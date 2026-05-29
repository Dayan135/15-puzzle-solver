#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cstddef>

// Thread-safe bounded blocking queue.
// push() blocks when the queue is at capacity.
// pop()  blocks when the queue is empty.
// shutdown() unblocks all waiters; subsequent pop() calls return false.
template<typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(std::size_t capacity)
        : capacity_(capacity), shutdown_(false) {}

    // Push an item.  Blocks if the queue is full.
    // Returns false if the queue has been shut down.
    bool push(T item) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_full_.wait(lk, [this] {
            return q_.size() < capacity_ || shutdown_;
        });
        if (shutdown_) return false;
        q_.push(std::move(item));
        lk.unlock();
        cv_not_empty_.notify_one();
        return true;
    }

    // Pop an item into `out`.  Blocks if the queue is empty.
    // Returns false when the queue is shut down AND empty.
    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_not_empty_.wait(lk, [this] {
            return !q_.empty() || shutdown_;
        });
        if (q_.empty()) return false; // shut down + drained
        out = std::move(q_.front());
        q_.pop();
        lk.unlock();
        cv_not_full_.notify_one();
        return true;
    }

    // Signal all threads to stop waiting.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            shutdown_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return q_.size();
    }

    bool is_shutdown() const {
        std::lock_guard<std::mutex> lk(mu_);
        return shutdown_;
    }

private:
    std::queue<T>           q_;
    mutable std::mutex      mu_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    std::size_t             capacity_;
    bool                    shutdown_;
};
