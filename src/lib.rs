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
//! between two actors sharing memory. To prevent this data race, it is not possible to:
//!
//! 1) Read the memory while it is being written to by the kernel, or write to the memory while it
//!    is being read by the kernel. This is exactly like Rust's aliasing rules: we can either allow
//!    both the kernel and this process to read a buffer, for example in the system call
//!    _write(2)_, we can temporarily give the kernel exclusive ownership one or more buffers when
//!    the kernel is going to write to them, or we can avoid sharing memory at all with the kernel,
//!    __but we cannot let either actor have mutable access while the other has any access at
//!    all.__
//! 2) Reclaim the memory while it is being read from or written to by the kernel. This is as
//!    simple as it sounds: we simply do not want the buffers to be used for other purposes, either
//!    by returning the memory to the heap, where it can be allocated simply so that the kernel can
//!    overwrite it when it is not supposed to, or it can corrupt stack variables.
//!
//! The term "kernel" does not necessarily have to be the other actor that the memory is shared
//! with; on Redox for example, the `io_uring` interface can work solely between regular userspace
//! processes. Additionally, although being a somewhat niche case, this can also be used for safe
//! wrappers protecting memory for DMA in device drivers, with a few additional restrictions
//! (regarding cache coherency) to make that work.
//!
//! This buffer sharing logic does unfortunately not play very well with the current asynchronous
//! ecosystem, where almost all I/O is done using regular borrowed slices, and references are
//! merely borrows which are cancellable _at any time_, even by leaking. This functions perfectly
//! when you use _synchronous_ (but non-blocking) system calls where either the process or the
//! kernel can execute at a time. In contrast, `io_uring` is _asynchronous_, meaning that the
//! kernel can read and write to buffers, _while our program is executing_. Therefore, a future
//! that locally stores an array, aliased by the kernel in `io_uring`, cannot stop the kernel from
//! using the memory again in any reasonable way, if the future were to be `Drop`ped, without
//! blocking indefinitely. What is even worse, is that futures can be leaked at any time, and
//! arrays allocated on the stack can also be dropped, when the memory is still in use by the
//! kernel, as a buffer to write data from e.g. a socket. If a (mutable) buffer of a stack is then
//! used for regular variables... arbitrary program corruption!
//!
//! What we need in order to solve these two complications, is some way to be able to mark a memory
//! region as both "borrowed by the kernel" (mutably or immutably), and "undroppable". Since the
//! Rust borrow checker is smart, any mutable reference with a lifetime that is shorter than
//! `'static`, can trivially be leaked, and the pointer can be used again. This rules out any
//! reference of lifetime `'a` that `'static` outlives, as those may be used again outside of the
//! borrow, potentially mutably. Immutable static references are however completely harmless, since
//! they cannot be dropped nor accessed mutably, and immutable aliasing is always permitted.
//!
//! Consequently, all buffers that are going to be used in safe code, must be owned. This either
//! means heap-allocated objects (since we can assume that the heap as a whole has the `'static`
//! lifetime, and allocations stay forever, until deallocated explicitly), buffer pools which
//! themselves have a guarding mechanism, and static references (both mutable and immutable). We
//! can however allow borrowed data as well, but because of the semantics around lifetimes, and the
//! very fact that the compiler has no idea that the kernel is also involved, that requires unsafe
//! code.
//!
//! Consider reading ["Mental experiments with
//! `io_uring`"](https://vorner.github.io/2019/11/03/io-uring-mental-experiments.html), and ["Notes
//! on `io-uring`"](https://without.boats/blog/io-uring/) for more information about these
//! challenges.
//!
//! # Interface
//! 
//! The main type that this crate provides, is the [`Guarded`] struct. [`Guarded`] is a wrapper
//! that encapsulates any stable pointer (which in safe code, can only be done for types that
//! implement [`StableDeref`]). The reason for [`StableDeref`] is because the memory must point to
//! the same buffer, and be independent of the location of the _pointer_ on the stack. Due to this,
//! a newtype that simply borrows an inner field cannot safely be stored in the [`Guarded`]
//! wrapper, but heap containers such as [`std::vec::Vec`] and [`std::boxed::Box`] implement that
//! trait.
//!
//! Once the wrapper is populated with an active guard, which typically happens when an `io_uring`
//! opcode is submitted, that will access memory, the wrapper will neither allow merely accessing
//! the memory, nor dropping it, until the guard successfully returns `true` after calling
//! [`Guard::try_release`]. It is then the responsibility of the guard, to make sure dynamically
//! that the memory is no longer aliased. There exist a fallible reclamation method, and a method
//! for trying to access the memory as well, both of which will succeed if the guard is either nonpresent,
//! or if [`Guard::try_release`] succeeds. The Drop impl will leak the memory entirely (by not
//! calling the `Drop` handler of the inner value), if the guard was not able to release the memory
//! when the buffer goes out of scope. It is thus highly advised to manually keep track of the
//! buffer, to prevent accidental leakage.
//!
//! There are a few corner-cases to these quite strict rules: while they are required for upholding
//! the reclamation invariant, the no-access restriction is not necessary for usages that only
//! _read_ memory. Consider a _write(2)_ system call for instance, which will never write to the
//! buffer, allowing the memory to be accessed _immutably_ while the guard is active, or mutably
//! once the guard is released. Because of this, there are marker types [`marker::Shared`] and
//! [`marker::Exclusive`], as generic parameters to [`Guarded`]. If [`marker::Shared`] is used, the
//! memory can always be accessed read-only.
//!
//! Static references are special in the way that they are guaranteed to never be dropped (at least
//! in Safe Rust), and they obviously include things like string literals and other static data.
//! However, a guard is still necessary for mutable references, since they could be written to when
//! they should not, if they were to be moved out from the wrapper.
//!
//! Furthermore, there are some types that handle their guarding mechanism themselves, unlike the
//! [`Guarded`] wrapper. The regular [`Guarded`] wrapper assumes that all data comes from the heap,
//! or some other global location that persists for the duration of the program, but is suboptimal
//! for custom allocators e.g. in buffer pools. To address this limitation, the [`Guardable`] trait
//! exists to abstracts the role of what the [`Guarded`] wrapper does, namely being able to insert
//! a guard, and then protect the memory until the guard can be released. For flexibilility,
//! __prefer `impl Guardable` rather than [`Guarded`], if possible__. This trait is implemented by
//! [`Guarded`], and the most notable example outside of this crate is `BufferSlice` from
//! [`redox-buffer-pool`](https://gitlab.redox-os.org/redox-os/redox-buffer-pool), that needs to
//! tell the buffer pool (which itself is dynamically allocated) that the pool has guarded memory,
//! to prevent it from deallocating that that particular slice, or the pool as a whole.
//!
//! This interface is conceptually roughly analoguous to the [`std::pin::Pin`] API offered by
//! `core` and `std`; there is a wrapper [`Guarded`] that encapsulates an pointer, and a trait for
//! types that can safely be released from the guard. However, there are is also a major
//! difference: __`Pin` types do not prevent memory reclamation on the stack by leaking, nor
//! mutable aliasing, _whatsoever_, making them unusable for `io_uring`. The only thing they
//! ensure, is that the inner type has to have a stable address (i.e. no moving out) before
//! dropping__. Meanwhile, `Guarded` __will enforce that one can share an address with another
//! process or hardware, that has no knowledge of when the buffer is getting reclaimed.__

#![cfg_attr(not(any(test, feature = "std")), no_std)]
//#![feature(maybe_uninit_ref)]

use core::mem::MaybeUninit;
use core::marker::PhantomData;
use core::{fmt, mem, ops};

pub extern crate stable_deref_trait;
pub use stable_deref_trait::StableDeref;

pub mod marker {
    /// A marker for memory regions that cannot be mutated, and thus lift the memory accessibility
    /// restriction, and only require safe memory reclamation.
    pub struct Shared {
        _not_creatable: (),
    }
    /// A marker for memory regions that _may_ be mutated, and hence require exclusive access.
    /// Using this marker, the [`Guarded`] wrapper will forbid simply accessing the memory, until
    /// the guard is released.
    pub struct Exclusive {
        _not_creatable: (),
    }

    mod private {
        pub trait Aliasable {}
    }
    impl private::Aliasable for Shared {}
}

/// A trait for guards, that decide whether memory can be deallocated, or whether it may be shared
/// with an actor outside of control from this process at all.
///
/// In an `io_uring` context, these will typically associate one (for exclusive or shared memory)
/// or more futures (for shared memory) with a memory region. In such a case, the guard will
/// succeed with the [`try_release`] operation, once it can prove that no pending operation is
/// accessing that memory.
///
/// # Safety
///
/// Safe code must obviously be able to use the guarding API soundly, and for memory to actually be
/// shared, the guard must ensure that the aliasing and reclamation invariants are upheld. However,
/// since the guard type is strictly dependent on the context where the guards are actually used,
/// this trait itself is not unsafe to implement, as it does not introduce any unsafe contract that
/// unsafe code can rely on. The code that shares the memory with the kernel will already require
/// unsafe code in order to be able to break any invariants regarding aliasing and data races in
/// the first place.
// TODO: Investigate whether this could be of use when it comes to concurrent memory reclamation,
// using epoch counts (`crossbeam-epoch`) or hazard pointers (`conc`), and whether there can be
// integration between those systems and this trait.
pub trait Guard {
    /// Try to release the guard, returning either `true` if the memory could be reclaimed, or `false` for failure.
    fn try_release(&self) -> bool;
}

/// A no-op guard, that cannot be initialized but still useful in type contexts. This the
/// recommended placeholder for types that do not need guarding.
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

/// A wrapper for types that can be "guarded", meaning that the memory they point to cannot be
/// safely reclaimed or even moved out, until the guard frees it. For exclusive-access buffers
/// (which are written to, in other words), the data is also completely unaccessible until the
/// guard is released.
///
/// This wrapper is in a way similar to [`std::pin::Pin`], in the sense that the inner value cannot
/// be moved out. The difference with this wrapper type, is that unlike Pin, which guarantees that
/// the address be stable until Drop, this type guarantees that the address be stable until the
/// guard can be released, that Drop leaks without the approval of the guard, and that the memory
/// cannot be aliased mutably with the kernel.
///
/// Rather than allowing arbitrary type to be guarded as with `Pin` (even though anyone can
/// trivially implement [`std::ops::Deref`] for a type that doesn't have a fixed location, like
/// simply taking the address of self), [`Guarded`] will also add additional restrictions that are
/// needed to initialize the wrapper _in safe code_:
///
/// * First, the data must be static (as in `T: 'static`), because a borrow with a shorter lifetime
/// `'a`, can be cancelled without ever calling the destructor, allowing the kernel process that
/// shares the memory, to overwrite memory locations that could then be used for regular variables,
/// or it could read undefined or secret data.
/// * Secondly, the data must implement [`std::ops::Deref`], since it does not make sense for
/// non-pointers to be guarded, as they obviously do not represent buffers in that case. Although
/// it is possible to Pin a `[u8; 1024]`-wrapper that simply `Deref`s by borrowing the on-stack
/// data, because Deref allows the type to dereference into a borrowed field of a struct that can
/// freely move on the stack, [`StableDeref`] is also required.
///
/// Note that while these traits must be implemented on the wrapped type to create a [`Guarded`] in
/// safe code, the wrapper can still be created using [`Guarded::new_unchecked`], where the caller
/// must ensure that the guarded type is only used in a context where no futures using
/// stack-borrowed memory are leaked, and that the `Deref` implementation is correct.
///
/// When the wrapper is guarded, the inner pointer can neither be accessed mutably, immutably for
/// exclusive memory regions, nor be moved out. There are however unsafe functions to bypass these
/// restrictions.
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
impl<G, T, M> Guarded<G, T, M>
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
    /// never be dropped without the guard allowing that, and never be reusable when leaked, those
    /// invariants must be upheld by the caller when calling this. __If the data is non-static and
    /// thus borrowed, great care must be taken to make sure that the entire program, and all
    /// possible executors with futures where this data may be borrowed from, always uphold the
    /// aliasing and relamation invariants.__
    pub unsafe fn new_unchecked(inner: T) -> Self {
        Self {
            inner: MaybeUninit::new(inner),
            guard: None,
            _marker: PhantomData,
        }
    }
    /// Wrap a type into a wrapper that can be a new guarded, which prevents the inner value from
    /// dropping or being moved out safely, until the guard releases itself.
    ///
    /// This associated function requires that `T` be both `'static` and [`StableDeref`]. The
    /// reason for this, is that `'static` ensures that the type does not contain any references,
    /// directly or indirectly, which when dropped allow the references to be reused again,
    /// potentially violating both the aliasing and reclamation invariants.
    ///
    /// Additionally, it must also implement [`StableDeref`], to make sure that the pointer also
    /// maintains an address that will never change. This requirement is as crucial as the guards
    /// and the dropping themselves. If a pointer is given to an external process or the kernel,
    /// that address must always be part of this memory region.
    pub fn new(inner: T) -> Self
    where
        T: 'static + StableDeref,
    {
        unsafe { Self::new_unchecked(inner) }
    }
    /// Apply a guard to the wrapper, which will make it impossible for the inner value to be moved
    /// out (unless using unsafe of course). Additionally, __the buffer will become temporarily
    /// unusable, since the memory may be written to if it is exclusive__. The memory will be
    /// leaked completely if the destructor is called when the guard cannot release the pointer.
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
    pub unsafe fn unguard_unchecked(&mut self) -> Option<G> {
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
    /// guard will not be released, and it is instead up to the caller to make sure that the rules
    /// of the specific guard type are followed.
    ///
    /// # Safety
    ///
    /// This is unsafe for the same reasons as with [`unguard_unchecked`]; as the guard is left
    /// untouched, it's completely up to the caller to ensure that the invariants required by the
    /// guard be upheld.
    ///
    /// [`unguard_unchecked`]: #method.unguard_unchecked
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
            Ok(unsafe { self.unguard_unchecked() })
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
        self.get_pointer_unchecked_ref().deref()
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
        self.get_pointer_unchecked_mut().deref_mut()
    }

    pub fn try_get_mut(&mut self) -> Option<&mut <T as ops::Deref>::Target>
    where
        T: ops::DerefMut,
    {
        if self.has_guard() {
            return None;
        }

        Some(unsafe { self.get_unchecked_mut() })
    }

    pub fn as_ptr(&self) -> *const <T as ops::Deref>::Target {
        unsafe { self.get_unchecked_ref() }
    }
    pub fn as_mut_ptr(&mut self) -> *mut <T as ops::Deref>::Target
    where
        T: ops::DerefMut,
    {
        unsafe { self.get_unchecked_mut() }
    }
    pub fn as_pointer_ptr(&self) -> *const T {
        unsafe { self.get_pointer_unchecked_ref() }
    }
    pub fn as_mut_pointer_ptr(&mut self) -> *mut T {
        unsafe { self.get_pointer_unchecked_mut() }
    }

    /// Unsafely obtain a reference to the pointer encapsulated by this wrapper.
    ///
    /// This is safe because one of the [`StableDeref`] contracts, is that the pointer must not
    /// be able to change the address via a method that takes a shared reference (i.e. not `&mut`).
    pub unsafe fn get_pointer_unchecked_ref(&self) -> &T {
        // TODO: #![feature(maybe_uninit_ref)]
        //unsafe { self.inner.assume_init_ref() }
        &*(&self.inner as *const MaybeUninit<T> as *const T)
    }
    pub fn try_get_pointer_ref(&self) -> Option<&T> {
        if self.has_guard() {
            return None;
        }

        Some(unsafe { self.get_pointer_unchecked_ref() })
    }
    /// Unsafely obtain a mutable reference to the pointer encapsulated by this wrapper.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it allows the pointer to trivially change its inner address,
    /// for example when Vec reallocates its space to expand the collection, thus violating the
    /// `StableDeref` contract.
    pub unsafe fn get_pointer_unchecked_mut(&mut self) -> &mut T {
        // TODO: #![feature(maybe_uninit_ref)]
        //self.inner.assume_init_mut()
        &mut *(&mut self.inner as *mut MaybeUninit<T> as *mut T)
    }
    pub fn try_get_pointer_mut(&mut self) -> Option<&mut T> {
        if self.has_guard() {
            return None;
        }
        Some(unsafe { self.get_pointer_unchecked_mut() })
    }
}
impl<G, T> Guarded<G, T, marker::Shared>
where
    G: Guard,
    T: ops::Deref,
{
    /// Obtain a reference to the pointer.
    pub fn get_pointer_ref(&self) -> &T {
        // SAFETY: This is safe since the Shared mode implies that the memory protected by this
        // guard cannot be written to by the external actor. Since the constructor for this type
        // requires the address to be stable, and not modified unless via a "&mut" method,
        // retrieving the pointer in this case is harmless.
        unsafe { self.get_pointer_unchecked_ref() }
    }

    /// Obtain a reference to the pointee.
    pub fn get_ref(&self) -> &<T as ops::Deref>::Target {
        // SAFETY: As with get_pointer_ref, this is safe since the wrapper is marked Shared, and
        // the address is already assumed to be stable.
        unsafe { self.get_unchecked_ref() }
    }
}
impl<G, T, M> fmt::Debug for Guarded<G, T, M>
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
            .field("mode", &core::any::type_name::<M>())
            .field("value", &*self)
            .field("guard", &GuardDbg(self.guard.as_ref()))
            .finish()
    }
}

impl<G, T, M> Drop for Guarded<G, T, M>
where
    G: Guard,
    T: ops::Deref,
{
    fn drop(&mut self) {
        if self.try_unguard().is_ok() {
            // Drop the inner value if the guard was able to be removed.
            drop(unsafe { self.uninitialize_inner() })
        } else {
            // Do nothing and leak the value otherwise.
        }
    }
}
impl<G, T> ops::Deref for Guarded<G, T, marker::Shared>
where
    G: Guard,
    T: ops::Deref,
{
    type Target = <T as ops::Deref>::Target;

    fn deref(&self) -> &Self::Target {
        self.get_ref()
    }
}
impl<G, T, U> AsRef<U> for Guarded<G, T, marker::Shared>
where
    G: Guard,
    T: ops::Deref<Target = U>,
    U: ?Sized,
{
    fn as_ref(&self) -> &U {
        self.get_ref()
    }
}
impl<G, T, M> From<T> for Guarded<G, T, M>
where
    G: Guard,
    T: ops::Deref + StableDeref + 'static,
{
    fn from(inner: T) -> Self {
        Self::new(inner)
    }
}
unsafe impl<G, T> StableDeref for Guarded<G, T, marker::Shared>
where
    G: Guard,
    T: ops::Deref,
{}

/// A trait for types that can be "guardable", meaning that they only leak on Drop unless they can
/// remove their guard, that their memory cannot be read from if the kernel may mutate it, and that
/// the memory cannot be written to when the kernel may read it.
///
/// # Safety
///
/// This trait is unsafe to implement, due to the following invariants that must be upheld:
///
/// * When invoking [`try_guard`], which takes a mutable reference, any pointers to self must not
/// be invalidated, which wouldn't be the case for e.g. a Vec that inserted a new item when
/// guarding. Additionally, the function must not in any way access the inner data that is being
/// guarded, since the futures will have references before even sending the guard, to that data.
/// * When dropping, the data _must not_ be reclaimed, until the guard that this type has received,
/// is successfully released.
/// * Once the guard is present, it must be impossible for either the kernel or this process to
/// mutably access the data, while it is being accessed simultaneously.
///
/// [`try_guard`]: #method.try_guard
pub unsafe trait Guardable<G> {
    /// Attempt to insert a guard into the guardable, if there wasn't already a guard inserted. If
    /// that were the case, error with the guard that wasn't able to be inserted.
    ///
    /// If the pointee of self (i.e. `<Self as ops::Deref>::Target`) implements [`Unguard`], then
    /// this method needs not be called.
    fn try_guard(&mut self, guard: G) -> Result<(), G>;
}

unsafe impl<G, T, M> Guardable<G> for Guarded<G, T, M>
where
    G: Guard,
    T: ops::Deref,
{
    fn try_guard(&mut self, guard: G) -> Result<(), G> {
        if self.has_guard() {
            return Err(guard);
        }
        Self::guard(self, guard);
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
        let _ = Guarded::<NoGuard, _, marker::Shared>::new(static_ref);
    }
}
