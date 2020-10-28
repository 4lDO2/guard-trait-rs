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
//!    all.__ (_aliasing invariant_)
//! 2) Reclaim the memory while it is being read from or written to by the kernel. This is as
//!    simple as it sounds: we simply do not want the buffers to be used for other purposes, either
//!    by returning the memory to the heap, where it can be allocated simply so that the kernel can
//!    overwrite it when it is not supposed to, or it can corrupt stack variables. (_reclamation
//!    invariant_)
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
//! kernel, as a buffer to write data from e.g. a socket. If a (mutable) buffer on the stack is
//! then reused later for regular variables... arbitrary program corruption!
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
//! The way `guard_trait` solves this, is by adding two simple traits: `Guarded` and `GuardedMut`.
//! `Guarded` is automatically implemented for every pointer type that implements `Deref`,
//! `StableDeref` and `'static`. Similarly, `GuardedMut` is implemented under the same conditions,
//! and provided that the pointer implements `DerefMut`. A consequence of this, is that nearly all
//! owned container types, such as `Arc`, `Box`, `Vec`, etc., all implement the traits, and can
//! thus be used with completion-based interfaces.
//!
//! For scenarios where it is impossible to ensure at the type level, that a certain pointer
//! follows the guard invariants, `AssertSafe` also exists, but is unsafe to initialize.
//!
//! Buffers can also be mapped in a self-referencial way, similar to how `owning-ref` works, using
//! `GuardedExt::map` and `GuardedMutExt::map_mut`. This is especially important when slice
//! indexing is needed, as the only way to limit the number of bytes to do I/O with, generally is
//! to shorten the slice.

#![deny(broken_intra_doc_links, missing_docs)]
#![cfg_attr(not(any(test, feature = "std")), no_std)]

use core::marker::PhantomData;
use core::ptr::NonNull;
use core::{fmt, ops};

// TODO: Perhaps consider basing everything on the Borrow trait, with StableBorrow.
pub extern crate stable_deref_trait;
pub use stable_deref_trait::StableDeref;

/// A trait for pointer types that uphold the guard invariants, namely that the pointer must be
/// owned, and that it must dereference into a stable address.
pub unsafe trait Guarded {
    /// The target pointee that this pointer may dereference into. There are no real restrictions
    /// to what this type can be. However, the user must not assume that simply because a `&Target`
    /// reference is protected, that references indirectly derived (via `Deref` and other traits)
    /// would also be protected.
    type Target: ?Sized;

    /// Borrow the pointee, into a fixed reference that can be sent directly and safely to e.g.
    /// memory-sharing completion-based I/O interfaces.
    ///
    /// Implementors of such interfaces must however take buffers by reference to maintain safety.
    fn borrow_guarded(&self) -> &Self::Target;
}
/// A trait for pointer types that uphold the guard invariants, and are able to dereference
/// mutably.
pub unsafe trait GuardedMut: Guarded {
    /// Borrow the pointee mutably, into a fixed reference that can be sent directly and safely to
    /// e.g. memory-sharing completion-based I/O interfaces.
    ///
    /// Implementors of such interfaces must however take buffers by reference to maintain safety.
    fn borrow_guarded_mut(&mut self) -> &mut Self::Target;
}

unsafe impl<T, U> Guarded for T
where
    T: ops::Deref<Target = U> + StableDeref + 'static,
    U: ?Sized,
{
    type Target = U;

    #[inline]
    fn borrow_guarded(&self) -> &U {
        &*self
    }
}
unsafe impl<T, U> GuardedMut for T
where
    T: ops::DerefMut<Target = U> + StableDeref + 'static,
    U: ?Sized,
{
    #[inline]
    fn borrow_guarded_mut(&mut self) -> &mut U {
        &mut *self
    }
}

/// A type for pointers that cannot uphold the necessary guard invariants at the type level, but
/// which can be assumed to behave properly by unsafe code.
#[repr(transparent)]
#[derive(Debug)]
pub struct AssertSafe<T> {
    inner: T,
}

impl<T> AssertSafe<T>
where
    T: ops::Deref,
{
    /// Wrap a general-purpose pointer into a wrapper that implements the `Guarded` (and
    /// potentially `GuardedMut`) traits, provided that the pointer upholds this invariants anyway.
    ///
    /// # Safety
    ///
    /// For the guard invariants to be upheld, the pointer must:
    ///
    /// * dereference into a stable location. This forbids types that implement `Deref` by
    ///   borrowing data that they own without a heap-based container in between;
    /// * be _owned_. Any type that has a shorter lifetime than `'static`, may have its borrow
    ///   cancelled _at any time_, with the original borrowed data accessible again.
    #[inline]
    pub unsafe fn new_unchecked(inner: T) -> Self {
        Self { inner }
    }
}
unsafe impl<T, U> Guarded for AssertSafe<T>
where
    T: ops::Deref<Target = U>,
    U: ?Sized,
{
    type Target = U;

    #[inline]
    fn borrow_guarded(&self) -> &Self::Target {
        &*self.inner
    }
}
unsafe impl<T, U> GuardedMut for AssertSafe<T>
where
    T: ops::Deref<Target = U> + ops::DerefMut,
    U: ?Sized,
{
    #[inline]
    fn borrow_guarded_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

/// A mapped guard, which contains a guarded owned pointer, and an immutable reference to that
/// pointer. It has had a one-time closure applied to it, but only the _output_ of the closure is
/// stored, not the closure itself. This is similar to how crates like `owning_ref` work.
pub struct Mapped<T, U>
where
    T: Guarded,
    U: ?Sized,
{
    // NOTE: It is important to make the distinction between the pointer and the pointee here. T is
    // a pointer type in this context, which is guaranteed to dereference into a stable address.
    // However, unlike with MappedMut, this can be trivially dereferenced at any time, since
    // immutable references allow multiple aliases.
    inner: T,
    // NOTE: NonNull gives us a covariant non-null pointer, which is valid for immutable
    // references. We cannot overwrite a subtype with a supertype without mutable access.
    mapped: NonNull<U>,
}
impl<T, U> Mapped<T, U>
where
    T: Guarded,
    U: ?Sized,
{
    /// Move the inner guarded pointer out from the `Mapped` wrapper, cancelling the temporary
    /// borrow.
    #[inline]
    pub fn into_original(self) -> T {
        self.inner
    }
    /// Get the original pointer, by immutable reference.
    #[inline]
    pub fn original_by_ref(&self) -> &T {
        &self.inner
    }
    /// Retrieve an immutable reference to the mapped data.
    #[inline]
    pub fn get_ref(&self) -> &U {
        unsafe { self.mapped.as_ref() }
    }
    /// Map the mapped wrapper again, converting `&U` to `&V`.
    #[inline]
    pub fn and_then<F, V>(self, f: F) -> Mapped<T, V>
    where
        F: FnOnce(&U) -> &V,
        V: ?Sized,
    {
        Mapped {
            mapped: f(self.get_ref()).into(),
            inner: self.inner,
        }
    }
}
impl<T, U> fmt::Debug for Mapped<T, U>
where
    T: Guarded,
    U: ?Sized + fmt::Debug,
{
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Mapped").field(&self.get_ref()).finish()
    }
}

unsafe impl<T, U> Guarded for Mapped<T, U>
where
    T: Guarded,
    U: ?Sized,
{
    type Target = U;

    #[inline]
    fn borrow_guarded(&self) -> &Self::Target {
        self.get_ref()
    }
}

/// A mapped guard, which contains a guarded owned pointer, and a mutable reference to that
/// pointer. It has had a one-time closure applied to it, but only the _output_ of the closure is
/// stored, not the closure itself. This is similar to how crates like `owning_ref` work.
pub struct MappedMut<T, U>
where
    T: GuardedMut,
    U: ?Sized,
{
    // NOTE: While we are allowed to take a reference to self, which in turn contains inner, as
    // well as inner itself, but we are not allowed to create a reference directly to the pointee
    // inner, since that would violate the no-aliasing rule enforced by Rust's mutable references.
    inner: T,
    mapped: NonNull<U>,

    // NOTE: We need to make this invariant, since we have a mutable reference, together with the
    // data that it points to. If it were to be covariant, then MappedMut<Subtype, Subtype>
    // would be trivially castable to MappedMut<Subtype, Supertype>. This is a problem since the
    // reference is mutable, allowing Subtype to be replaced with Supertype.
    //
    // The regular Mapped wrapper only deals with immutable references, and is thus covariant.
    //
    // See [the nomicon](https://doc.rust-lang.org/nomicon/subtyping.html).
    _invariance: PhantomData<*mut U>,
}

impl<T, U> MappedMut<T, U>
where
    T: GuardedMut,
    U: ?Sized,
{
    /// Move out the original pointer from the mapped guard, hence cancelling the temporary borrow.
    #[inline]
    pub fn into_original(self) -> T {
        self.inner
    }
    /// Convert `MappedMut<T, U>` to `Mapped<T, U>`.
    #[inline]
    pub fn into_immutable(this: MappedMut<T, U>) -> Mapped<T, U> {
        Mapped {
            inner: this.inner,
            mapped: this.mapped,
        }
    }
    /// Get an immutable reference to the mapped data.
    #[inline]
    pub fn get_ref(&self) -> &U {
        unsafe { self.mapped.as_ref() }
    }
    /// Get a mutable reference to the mapped data.
    #[inline]
    pub fn get_mut(&mut self) -> &mut U {
        unsafe { self.mapped.as_mut() }
    }
    /// Map the mapped reference again, converting `&mut U` to `&mut V`.
    #[inline]
    pub fn and_then<F, V>(mut self, f: F) -> MappedMut<T, V>
    where
        F: FnOnce(&mut U) -> &mut V,
        V: ?Sized,
    {
        MappedMut {
            mapped: f(self.get_mut()).into(),
            inner: self.inner,

            _invariance: PhantomData,
        }
    }
}

unsafe impl<T, U> Guarded for MappedMut<T, U>
where
    T: GuardedMut,
    U: ?Sized,
{
    type Target = U;

    #[inline]
    fn borrow_guarded(&self) -> &Self::Target {
        self.get_ref()
    }
}
unsafe impl<T, U> GuardedMut for MappedMut<T, U>
where
    T: GuardedMut,
    U: ?Sized,
{
    #[inline]
    fn borrow_guarded_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

// TODO: Dynamically tracked "anchors" that allow referenced with lifetimes, with the cost of some
// extra runtime tracking, to prevent dropped or leaked referenced from doing harm. This would be
// similar to what the old API did.

// TODO: Support mapped types that map into a borrowing _object_, rather than a plain reference.
// It's not entirely clear what owning_ref does wrong there, but consider [this
// issue](https://github.com/Kimundi/owning-ref-rs/issues/49).

mod private {
    pub trait Sealed {}
}
/// An extension trait for convenience methods, that is automatically implemented for all
/// [`Guarded`] types.
pub trait GuardedExt: private::Sealed + Guarded + Sized {
    /// Apply a function to the pointee, creating a new guarded type that dereferences into the
    /// result of that function.
    ///
    /// The closure is only evaluated once, and the resulting wrapper will only store one
    /// null-optimizable additional word, for the reference.
    #[inline]
    fn map<F, T>(this: Self, f: F) -> Mapped<Self, T>
    where
        F: FnOnce(&<Self as Guarded>::Target) -> &T,
        T: ?Sized,
    {
        Mapped {
            mapped: f(this.borrow_guarded()).into(),
            inner: this,
        }
    }
}
/// An extension trait for convenience methods, that is automatically implemented for all
/// [`GuardedMut`] types.
pub trait GuardedMutExt: private::Sealed + GuardedMut + Sized {
    /// Apply a function to the pointee, creating a new guarded type that dereferences into the
    /// result of that function.
    ///
    /// This is the mutable version of [`GuardedExt::map`]. Because of this mutability, the
    /// original pointer cannot be accessed until it is completely moved out of the wrapper.
    #[inline]
    fn map_mut<F, T>(mut this: Self, f: F) -> MappedMut<Self, T>
    where
        F: FnOnce(&mut <Self as Guarded>::Target) -> &mut T,
        T: ?Sized,
    {
        MappedMut {
            mapped: f(this.borrow_guarded_mut()).into(),
            inner: this,
            _invariance: PhantomData,
        }
    }
}
impl<T> private::Sealed for T
where
    T: Guarded + Sized,
{
}
impl<T> GuardedExt for T
where
    T: Guarded + Sized,
{
}
impl<T> GuardedMutExt for T
where
    T: GuardedMut + Sized,
{
}

// TODO: Perhaps an additional extension trait that allows indexing and slicing pointers to slice
// types?

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_types_implement_guarded() {
        use std::rc::Rc;
        use std::sync::Arc;

        fn does_impl_guarded<T: Guarded>() {}
        fn does_impl_guarded_mut<T: GuardedMut>() {}

        does_impl_guarded::<Arc<[u8]>>();
        does_impl_guarded::<Rc<[u8]>>();
        does_impl_guarded::<Box<[u8]>>();
        does_impl_guarded::<Vec<u8>>();
        does_impl_guarded::<&'static [u8]>();
        does_impl_guarded::<&'static mut [u8]>();
        does_impl_guarded::<&'static str>();
        does_impl_guarded::<String>();

        does_impl_guarded_mut::<Box<[u8]>>();
        does_impl_guarded_mut::<Vec<u8>>();
        does_impl_guarded::<&'static mut [u8]>();
    }

    #[test]
    fn mapped() {
        let mut buf = vec! [0x00_u8; 256];

        let sub_buf = GuardedMutExt::map_mut(buf, |buf: &mut [u8]| -> &mut [u8] {
            &mut buf[128..]
        });
        let mut subsub_buf = sub_buf.and_then(|buf| &mut buf[..64]);

        for byte in subsub_buf.borrow_guarded_mut() {
            *byte = 0xFF;
        }
        buf = subsub_buf.into_original();

        assert!(buf[..128].iter().copied().all(|byte| byte == 0x00));
        assert!(buf[128..192].iter().copied().all(|byte| byte == 0xFF));
        assert!(buf[192..].iter().copied().all(|byte| byte == 0x00));

    }
}
