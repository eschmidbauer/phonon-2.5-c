/*  threadpool.h – Minimal pthread-based worker pool for parallel-for.
 *
 *  Design:
 *    • N-1 background worker threads + the calling thread participates as
 *      tid=0. No external deps beyond POSIX pthreads (libSystem on macOS,
 *      glibc on Linux).
 *    • Fork-join semantics: p25_parallel() returns only after all workers
 *      finish the current job.
 *    • A process-wide singleton pool is provided via p25_pool() and is
 *      lazy-initialised on first use. Thread count: clamp(P25_THREADS env
 *      var, 1, P25_MAX_THREADS) defaulting to NPROCS_ONLN.
 *    • Callers serving many concurrent sessions can additionally create
 *      independent pools with p25_pool_create() and select one per thread
 *      with p25_set_pool() — p25_pool() returns the thread-local override
 *      when set. This avoids the p25_parallel dispatch-mutex serialising
 *      cross-session encoder calls.
 */
#ifndef P25_THREADPOOL_H
#define P25_THREADPOOL_H

typedef struct ThreadPool ThreadPool;

/* Signature of work functions passed to p25_parallel.
 * `tid` is in [0, nthreads), `nthreads` is the total worker count
 * (including the calling thread). */
typedef void (*ParallelFn)(int tid, int nthreads, void *arg);

/* Return the pool to use on the calling thread. If p25_set_pool() has been
 * called on this thread with a non-NULL pool, that pool is returned;
 * otherwise the lazy-initialised process-wide singleton is returned. */
ThreadPool *p25_pool(void);

/* Create an independent pool with the given thread count (clamped to
 * [1, P25_MAX_THREADS]). The caller owns the returned pool and must call
 * p25_pool_destroy() when done. The singleton p25_pool() is unaffected. */
ThreadPool *p25_pool_create(int n_threads);

/* Stop workers and free `p`. Safe to pass NULL. Do NOT call on the pool
 * returned by p25_pool() — that one lives until program exit. */
void        p25_pool_destroy(ThreadPool *p);

/* Override the pool returned by p25_pool() on the calling thread. Pass NULL
 * to clear the override and fall back to the global singleton. The pool
 * pointer is stored in thread-local storage and must outlive every
 * subsequent p25_pool()/p25_parallel() call on this thread until cleared. */
void        p25_set_pool(ThreadPool *p);

/* Number of workers in the pool (including the calling thread). */
int p25_pool_nthreads(ThreadPool *p);

/* Dispatch `fn` to all workers. The calling thread runs as tid=0 in parallel
 * with the pool workers (tid=1..N-1). Returns after all finish. */
void p25_parallel(ThreadPool *p, ParallelFn fn, void *arg);

#endif /* P25_THREADPOOL_H */
