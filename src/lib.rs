//! This crate provides a guarding mechanism for memory, with an interface that is in some ways
//! similar to `core::pin::Pin`.
//!
//! # Motivation
//!
//! What this crate attempts to solve, is the problem that data races can occur for memory that is
//! shared with another process or the kernel (via `io_uring` for instance). If the memory is still
//! shared when the original thread continues to execute after a system call for example, the
//! original buffer can still be accessed while the system call allows the handler to keep using
//! the memory. This does not happen with traditional blocking syscalls; the kernel will only
//! access the memory during the syscall, when the process cannot temporarily do anything else.
//!
//! However, for more advanced asynchronous interfaces such as `io_uring` that never copy memory,
//! the memory can still be used once the system call has started, which can lead to data races
//! between two actors sharing memory. This is possible if an old memory range that the other actor
//! still thinks it can use, is then used for different purpose by this process. Additionally,
//! although being a niche case, this can also be used for safe wrappers protecting memory for DMA
//! in device drivers, with a few additional restrictions to make that work.
//!
//! This does unfortunately not play very well with the current asynchronous ecosystem, where
//! almost all I/O is done using regular slices, and references are merely borrows which are
//! cancellable _at any time_, even by leaking. So, a future that locally stores an array, which is
//! aliased by the kernel in `io_uring`, cannot stop the kernel from using the memory again in any
//! reasonable way, without blocking indefinitely. What is even worse, is that futures can be
//! leaked at any time, and arrays allocated on the stack can also be dropped, when the memory is
//! still in use by the kernel, as a buffer to write data from a socket into.
//!
//! # Interface
//! 
//! The main type that this crate provides, is the [`Guarded`] struct. [`Guarded`] is a wrapper
//! that encapsulates any stable pointer (anything that implements [`StableDeref`]), and does not
//! allow dropping, until the guard successfully returns `true` in [`Guard::try_release`]. There
//! exists a fallible reclamation method as well, and the Drop impl will leak the memory entirely,
//! if the guard was not able to release the memory at that point.
//!
//! There are a few corner-cases to this: for example, static references are guaranteed to never be
//! dropped (at least in Safe Rust), which also includes things like string literals and other
//! static data. To make APIs more ergonomic, there is also a trait for types that can trivially be
//! moved out from a guard, because they can never be reclaimed, and are always valid. For this,
//! there exists a trait [`Unguard`], that is only implemented for static references pointing to
//! static data. Types that implement [`Unguard`] can always safely be moved out from [`Guarded`],
//! without needing the guard to agree.
//!
//! Additionally, there exist some types that handle their guarding mechanism themselves. The
//! [`Guardable`] trait exists for this, which represents types that can insert a guard, and follow
//! the same rules as [`Guarded`]. __Because of this, prefer `impl Guardable` rather than
//! [`Guarded`], if possible. This trait is implemented by [`Guarded`], and the most notable
//! example outside of this crate is `BufferSlice` from
//! [`redox-buffer-pool`](https://gitlab.redox-os.org/redox-os/redox-buffer-pool), that needs to
//! tell the buffer pool that the pool has guarded memory, to prevent it from deallocating that
//! memory, that particular slice.
//!
//! This interface is roughly analoguous to the `Pin` API offered by `core` and `std`; there is a
//! wrapper `Guarded` that encapsulates an pointer, and a trait for types that can safely be
//! released from the guard. However, there are is also a major difference: __`Pin` types do not
//! prevent memory reallocation _whatsoever_, and the only thing they ensure, is that the inner
//! type has to have a stable address (i.e. no moving out) before dropping __. Meanwhile, `Guarded`
//! __will enforce that one can share an address with another process or hardware, that has no
//! knowledge of when the buffer is getting reclaimed.__

#![cfg_attr(not(any(test, feature = "std")), no_std)]
//#![feature(maybe_uninit_ref)]

use core::mem::MaybeUninit;
use core::marker::PhantomData;
use core::{fmt, mem, ops};

pub extern crate stable_deref_trait;
pub use stable_deref_trait::StableDeref;

/// A trait for guards, that decide whether memory can be deallocated, or whether it may be shared
/// with an actor outside of control from this process at all.
///
/// An example of such a guard type, is `CommandFutureGuard` from
/// [`redox-iou`](https://gitlab.redox-os.org/redox-os/redox-iou), that associates a guarded piece
/// of memory, with a future representing a pending submission and completion from a Redox
/// `io_uring`. As the `io_uring` interface is zero-copy and allows the kernel (or another process,
/// for redox) to use the memory that was part of the system call, until the completion
// TODO: Investigate whether this is of use when it comes to concurrent memory reclamation, using
// epochs (`crossbeam-epoch`) or hazard pointers (`conc`), and whether there can be integration
// between those and this trait.
pub trait Guard {
    /// Try to release the guard, returning either true for success or false for failure.
    fn try_release(&self) -> bool;

    /// A function that allows the guard to tell the wrapper whether the guarded memory could be
    /// written to for the lifetime of the guard. If this is false, which is possible in some
    /// scenarios (e.g. for a future that gives a read-only buffer to the kernel as part of a
    /// syscall), read-only access to the guarded may be granted.
    fn external_may_write(&self) -> bool { true }
}

/// A no-op guard, that cannot be initialized but still useful in type contexts. This the
/// recommended placeholder for types that do not need guarding, i.e. [`Unguard`] types.
///
/// This will be replaced with the never type, once that is stabilized.
#[derive(Debug)]
pub enum NoGuard {}

impl Guard for NoGuard {
    fn try_release(&self) -> bool {
        unreachable!("NoGuard cannot be initialized")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TryUnguardError;

impl fmt::Display for TryUnguardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to remove guard: guard itself could not released")
    }
}
#[cfg(feature = "std")]
impl std::error::Error for TryUnguardError {}

/// A wrapper for types that can be "guarded", meaning that the memory their reference cannot be
/// safely reclaimed or even moved out, until the guard frees it.
///
/// This wrapper is in a way similar to [`std::pin::Pin`], in the sense that the inner value cannot
/// be moved out. The difference with this wrapper type, is that unlike Pin, which guarantees that
/// the address be stable until Drop, this type guarantees that the address be stable until the
/// guard can be released, and that Drop leaks without the approval of the guard.
///
/// Thus, rather than allowing arbitrary type to be guarded as with `Pin` (even though anyone can
/// trivially implement [`std::ops::Deref`] for a type that doesn't have a fixed location, like
/// simply taking the address of self), Guarded will also add additional restrictions:
///
/// * First, the data must be static, because the data must not be removed even if the guard is
/// leaked. A leaked borrow is the same as a dropped borrow, but without the destructor run, and
/// even if the destructor were run, the data couldn't be leaked since it was borrowed from
/// somewhere else. Note that static references are allowed here though, since they are never ever
/// dropped.
/// * Secondly, the data must implement [`std::ops::Deref`], since it doesn't make sense for
/// non-pointers to be guarded. Sure, I could Pin a `[u8; 1024]` wrapper that simply `Deref`ed by
/// borrowing the on-stack data. Deref only allows the type to act as smart pointer, and to work
/// around the aforementioned limitation of Deref, [`StableDeref`] is also required.
///
/// Just like with `Pin`, `Guarded` will not allow mutable access to the pointer, but only to the
/// pointee. This is to prevent collections like `Vec` from reallocating (which is really easy if
/// one retrieves a mutable reference to a Vec), where the address can freely change. However,
/// unlike Pin that only allows the address to change after the destructor is run, this wrapper
/// will allow the pointer to be moved out dynamically, provided that the guard can ensure the data
/// is no longer shared.
pub struct Guarded<G, T, M>
where
    G: Guard,
    T: ops::Deref,
{
    // TODO: Allow layout optimization by not using MaybeUninit.
    inner: MaybeUninit<T>,
    guard: Option<G>,
    _marker: PhantomData<M>,
}
impl<G, T> Guarded<G, T>
where
    G: Guard,
    T: ops::Deref,
{
    /// Wrap a type into a wrapper that can be a new guarded, which prevents the inner value from
    /// dropping or being moved out safely, until the guard releases itself.
    ///
    /// # Safety
    ///
    /// Since both safe and unsafe code can assume that the pointee will never change its address,
    /// never be dropped without the guard allowing that, and never be reclaimable when leaked.
    /// Hence, it is allowed to pass a reference here, so long as one can ensure those invariants
    /// are upheld.
    pub unsafe fn new_unchecked(inner: T) -> Self {
        Self {
            inner: MaybeUninit::new(inner),
            guard: None,
        }
    }
    /// Wrap a type into a wrapper that can be a new guarded, which prevents the inner value from
    /// dropping or being moved out safely, until the guard releases itself.
    ///
    /// This associated function requires that `T` be both `'static` and [`StableDeref`]. The
    /// reason for this, is that `'static` ensures that the type does not contain any references,
    /// directly or indirectly, which when dropped again allow the references to be reused, which
    /// would violate guarding invariant.
    ///
    /// Additionally, it must also implement [`StableDeref`], to make sure that the pointer also
    /// maintains an address that will never change. This requirement is as crucial as the guards
    /// and the dropping themselves.
    pub fn new(inner: T) -> Self
    where
        T: 'static + StableDeref,
    {
        unsafe { Self::new_unchecked(inner) }
    }
    /// Apply a guard to the wrapper, which will make it impossible for the inner value to be moved
    /// out (unless using unsafe of course). Additionally, __the buffer will become temporarily
    /// unusable, since the memory may be written to__. The memory will be leaked completely if the
    /// destructor is called when the guard cannot release the pointer.
    pub fn guard(&mut self, guard: G) {
        assert!(self.guard.is_none());
        self.guard = Some(guard);
    }
    /// Query whether this wrapper contains an active guard.
    pub fn has_guard(&self) -> bool {
        self.guard.is_some()
    }
    /// Remove the guard, bypassing any safety guarantees provided by this wrapper.
    ///
    /// # Safety
    ///
    /// Since this removes the guard, this will allow the memory to be read or written to, or be
    /// reclaimed when some other entity could be using it simultaneously. For this not to lead to
    /// UB, the caller must not reclaim the memory owned here unless it can absolutely be sure that
    /// the guard is no longer needed.
    pub unsafe fn force_unguard(&mut self) -> Option<G> {
        self.guard.take()
    }

    /// Try to remove the guard together with the inner value, returning the wrapper again if the
    /// guard was not able to be safely released.
    pub fn try_into_inner(mut self) -> Result<(T, Option<G>), Self> {
        match self.try_unguard() {
            Ok(guard_opt) => Ok({
                let inner = unsafe { self.uninitialize_inner() };
                mem::forget(self);
                (inner, guard_opt)
            }),
            Err(_) => Err(self),
        }
    }
    /// Move out the inner value of this wrapper, together with the guard if there was one. The
    /// guard will not be released, and is instead up to the caller.
    ///
    /// # Safety
    ///
    /// This is unsafe for the same reasons as with [`force_unguard`]; as the guard is left
    /// untouched, it's completely up to the caller to ensure that the invariants required by the
    /// guard be upheld.
    ///
    /// [`force_unguard`]: #method.force_unguard
    pub unsafe fn into_inner_unchecked(mut self) -> (T, Option<G>) {
        let guard = self.guard.take();
        let inner = self.uninitialize_inner();
        mem::forget(self);
        (inner, guard)
    }

    /// Try removing the guard in-place, failing if the guard returned false when invoking
    /// [`Guard::try_release`].
    pub fn try_unguard(&mut self) -> Result<Option<G>, TryUnguardError> {
        let guard = match self.guard.as_ref() {
            Some(g) => g,
            None => return Ok(None),
        };

        if guard.try_release() {
            Ok(unsafe { self.force_unguard() })
        } else {
            Err(TryUnguardError)
        }
    }
    unsafe fn uninitialize_inner(&mut self) -> T {
        // self.inner.read() - TODO: use this when #![feature(maybe_uninit_extra)]
        mem::replace(&mut self.inner, MaybeUninit::uninit()).assume_init()
    }
    /// Obtain a reference to the pointee of the value protected by this guard.
    ///
    /// The address of this reference is guaranteed to be the same so long as the pointer stay
    /// within the guard. While this contract is true for the reference of the pointee itself, it
    /// is _not_ true for the pointee of the pointee in case the pointee of this guard is itself a
    /// pointer. So while you can safely have a `Guarded<Box<Vec<u8>>>`, only the `&Vec<u8>`
    /// reference will be valid, not the data within the vector.
    ///
    /// # Safety
    ///
    /// Since the data is allowed to change arbitrarily by external non-controllable processes,
    /// this may not be safe in all situations __if the guard is there__. However, for system calls
    /// that only read the buffer (e.g. _write(2)_), this is safe, however this depends on the
    /// context.
    pub unsafe fn get_unchecked_ref(&self) -> &<T as ops::Deref>::Target {
        self.get_pointer_ref_unchecked().deref()
    }
    /// Attempt to obtain a shared reference to the inner data, returning None if a guard is
    /// present.
    pub fn try_get_ref(&self) -> Option<&<T as ops::Deref>::Target> {
        if self.has_guard() {
            return None;
        }

        Some(unsafe { self.get_unchecked_ref() })
    }
    /// Obtain a mutable reference to the pointee of the value protected by this guard.
    ///
    /// See [`get_ref`] for a more detailed explanation of the guarantees of this method.
    pub unsafe fn get_unchecked_mut(&mut self) -> &mut <T as ops::Deref>::Target
    where
        T: ops::DerefMut,
    {
        unsafe { self.get_pointer_mut() }.deref_mut()
    }

    /// Unsafely obtain a reference to the pointer encapsulated by this wrapper.
    ///
    /// This is safe because one of the [`StableDeref`] contracts, is that the pointer must not
    /// be able to change the address via a method that takes a shared reference (i.e. not `&mut`).
    pub unsafe fn get_pointer_ref_unchecked(&self) -> &T {
        unsafe { self.inner.assume_init_ref() }
    }
    /// Unsafely obtain a mutable reference to the pointer encapsulated by this wrapper.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it allows the pointer to trivially change its inner address,
    /// for example when Vec reallocates its space to expand the collection, thus violating the
    /// `StableDeref` contract.
    pub unsafe fn get_pointer_mut(&mut self) -> &mut T {
        self.inner.assume_init_mut()
    }
}
impl<G, T> fmt::Debug for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        struct GuardDbg<'a, H>(Option<&'a H>);

        impl<'a, H> fmt::Debug for GuardDbg<'a, H> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                match self.0 {
                    Some(_) => write!(
                        f,
                        "[guarded using guard type `{}`]",
                        std::any::type_name::<H>()
                    ),
                    None => write!(f, "[no guard]"),
                }
            }
        }

        struct OpaqueDbg;

        impl fmt::Debug for OpaqueDbg {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "[temporarily unreadable due to guard]")
            }
        }

        f.debug_struct("Guarded")
            .field("value", &*self)
            .field("guard", &GuardDbg(self.guard.as_ref()))
            .finish()
    }
}

pub enum Exclusive {}
pub enum Sharable {}

impl<G, T> Drop for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref,
{
    fn drop(&mut self) {
        if self.try_unguard().is_ok() {
            // Drop the inner value if the guard was able to be removed.
            drop(unsafe { self.uninitialize_inner() })
        } else {
            // No nothing and leak the value otherwise.
        }
    }
}
impl<G, T> ops::Deref for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref,
{
    type Target = <T as ops::Deref>::Target;

    fn deref(&self) -> &Self::Target {
        unsafe { self.inner.get_ref() }
    }
}
impl<G, T> ops::DerefMut for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref + ops::DerefMut,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.inner.get_mut() }
    }
}
impl<G, T, U> AsRef<U> for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref<Target = U>,
    U: ?Sized,
{
    fn as_ref(&self) -> &U {
        &*self
    }
}
impl<G, T, U> AsMut<U> for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref<Target = U> + ops::DerefMut,
    U: ?Sized,
{
    fn as_mut(&mut self) -> &mut U {
        &mut *self
    }
}
impl<G, T> From<T> for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref + StableDeref + 'static,
{
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}
unsafe impl<G, T> StableDeref for Guarded<G, T>
where
    G: Guard,
    T: ops::Deref,
{}

/// A trait for types that can be "guardable", meaning that they won't do anything on Drop unless
/// they can remove their guard.
///
/// # Safety
///
/// This trait is unsafe to implement, due to the following invariants that must be upheld:
///
/// * When invoking [`try_guard`], any pointers to self must not be invalidated, which wouldn't be the
///   case for e.g. a Vec that inserted a new item when guarding. Additionally, the function must not
///   in any way access the inner data that is being guarded, since the futures will have references
///   before even sending the guard, to that data.
/// * When dropping, the data _must not_ be reclaimed, until the guard that this type has received,
///   is successfully released.
/// * The inner data must implement [`std::pin::Unpin`] or follow equivalent rules; if this
///   guardable is leaked, the data must no longer be accessible (this rules out data on the stack).
///
/// [`try_guard`]: #method.try_guard
pub unsafe trait Guardable<G>: ops::Deref {
    /// Attempt to insert a guard into the guardable, if there wasn't already a guard inserted. If
    /// that were the case, error with the guard that wasn't able to be inserted.
    ///
    /// If the pointee of self (i.e. `<Self as ops::Deref>::Target`) implements [`Unguard`], then
    /// this method needs not be called.
    fn try_guard(&mut self, guard: G) -> Result<(), G>;
}

unsafe impl<G, T> Guardable<G> for Guarded<G, T>
where
    G: Guard,
    T: StableDeref + 'static + ops::Deref,
{
    fn try_guard(&mut self, guard: G) -> Result<(), G> {
        if self.has_guard() {
            return Err(guard);
        }
        Self::guard(self, guard);
        Ok(())
    }
}

pub unsafe trait Unguard: ops::Deref {}

// The following implementations exist because a static reference which pointee also is static,
// cannot be dropped at all.
unsafe impl<T> Unguard for &'static T
where
    T: 'static + ?Sized,
{}
unsafe impl<T> Unguard for &'static mut T
where
    T: 'static + ?Sized,
{}

unsafe impl<G, T> Guardable<G> for T
where
    T: ops::Deref + ?Sized + Unguard,
{
    fn try_guard(&mut self, _guard: G) -> Result<(), G> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STATIC_DATA: [u8; 13] = *b"Hello, world!";

    #[test]
    fn safely_create_guarded_to_static_ref() {
        let static_ref: &'static [u8] = &STATIC_DATA;
        let _ = Guarded::<NoGuard, _>::new(static_ref);
    }
}
