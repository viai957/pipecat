use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A synchronous, typed event handler.
pub type EventHandler<T> = Box<dyn Fn(&T) + Send + Sync>;

/// A unique handle returned when registering an event handler, used for removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandlerId(u64);

/// A synchronous event emitter that dispatches events by string name.
///
/// This emitter does NOT require an async runtime. Handlers are invoked
/// synchronously in registration order when `emit` is called.
pub struct EventEmitter<T: 'static> {
    handlers: Arc<Mutex<EventEmitterInner<T>>>,
}

struct EventEmitterInner<T: 'static> {
    next_id: u64,
    handlers: HashMap<String, Vec<(HandlerId, EventHandler<T>)>>,
}

impl<T: 'static> EventEmitter<T> {
    /// Create a new event emitter with no registered handlers.
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(Mutex::new(EventEmitterInner {
                next_id: 1,
                handlers: HashMap::new(),
            })),
        }
    }

    /// Register a handler for the given event name.
    ///
    /// Returns a `HandlerId` that can be used to remove the handler later.
    pub fn on<F>(&self, event: &str, handler: F) -> HandlerId
    where
        F: Fn(&T) + Send + Sync + 'static,
    {
        let mut inner = self.handlers.lock().unwrap();
        let id = HandlerId(inner.next_id);
        inner.next_id += 1;
        inner
            .handlers
            .entry(event.to_string())
            .or_default()
            .push((id, Box::new(handler)));
        id
    }

    /// Remove a previously registered handler by its ID.
    ///
    /// Returns `true` if the handler was found and removed.
    pub fn off(&self, handler_id: HandlerId) -> bool {
        let mut inner = self.handlers.lock().unwrap();
        for handlers in inner.handlers.values_mut() {
            if let Some(pos) = handlers.iter().position(|(id, _)| *id == handler_id) {
                let _ = handlers.remove(pos);
                return true;
            }
        }
        false
    }

    /// Emit an event, invoking all handlers registered for `event` in order.
    pub fn emit(&self, event: &str, data: &T) {
        let inner = self.handlers.lock().unwrap();
        if let Some(handlers) = inner.handlers.get(event) {
            for (_, handler) in handlers {
                handler(data);
            }
        }
    }

    /// Returns the number of handlers registered for a given event.
    pub fn handler_count(&self, event: &str) -> usize {
        let inner = self.handlers.lock().unwrap();
        inner
            .handlers
            .get(event)
            .map_or(0, |handlers| handlers.len())
    }
}

impl<T: 'static> Default for EventEmitter<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> Clone for EventEmitter<T> {
    fn clone(&self) -> Self {
        Self {
            handlers: Arc::clone(&self.handlers),
        }
    }
}

// Safety: EventEmitter is Send + Sync because inner state is behind Arc<Mutex<..>>
// and EventHandler requires Send + Sync.
unsafe impl<T: 'static> Send for EventEmitter<T> {}
unsafe impl<T: 'static> Sync for EventEmitter<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn emit_calls_handler() {
        let emitter = EventEmitter::<String>::new();
        let count = Arc::new(AtomicU32::new(0));
        let count2 = Arc::clone(&count);

        emitter.on("test", move |_data| {
            count2.fetch_add(1, Ordering::SeqCst);
        });

        emitter.emit("test", &"hello".to_string());
        assert_eq!(count.load(Ordering::SeqCst), 1);

        emitter.emit("test", &"world".to_string());
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn emit_unknown_event_is_noop() {
        let emitter = EventEmitter::<u32>::new();
        emitter.emit("nonexistent", &42); // should not panic
    }

    #[test]
    fn multiple_handlers_called_in_order() {
        let emitter = EventEmitter::<()>::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let o1 = Arc::clone(&order);
        emitter.on("ev", move |_| o1.lock().unwrap().push(1));

        let o2 = Arc::clone(&order);
        emitter.on("ev", move |_| o2.lock().unwrap().push(2));

        let o3 = Arc::clone(&order);
        emitter.on("ev", move |_| o3.lock().unwrap().push(3));

        emitter.emit("ev", &());
        assert_eq!(*order.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn off_removes_handler() {
        let emitter = EventEmitter::<()>::new();
        let count = Arc::new(AtomicU32::new(0));
        let count2 = Arc::clone(&count);

        let id = emitter.on("ev", move |_| {
            count2.fetch_add(1, Ordering::SeqCst);
        });

        emitter.emit("ev", &());
        assert_eq!(count.load(Ordering::SeqCst), 1);

        assert!(emitter.off(id));
        emitter.emit("ev", &());
        assert_eq!(count.load(Ordering::SeqCst), 1); // unchanged
    }

    #[test]
    fn off_returns_false_for_unknown() {
        let emitter = EventEmitter::<()>::new();
        assert!(!emitter.off(HandlerId(9999)));
    }

    #[test]
    fn handler_count() {
        let emitter = EventEmitter::<()>::new();
        assert_eq!(emitter.handler_count("ev"), 0);

        emitter.on("ev", |_| {});
        assert_eq!(emitter.handler_count("ev"), 1);

        emitter.on("ev", |_| {});
        assert_eq!(emitter.handler_count("ev"), 2);

        assert_eq!(emitter.handler_count("other"), 0);
    }

    #[test]
    fn handler_receives_data() {
        let emitter = EventEmitter::<i32>::new();
        let received = Arc::new(Mutex::new(0i32));
        let r = Arc::clone(&received);

        emitter.on("ev", move |val| {
            *r.lock().unwrap() = *val;
        });

        emitter.emit("ev", &42);
        assert_eq!(*received.lock().unwrap(), 42);
    }

    #[test]
    fn clone_shares_handlers() {
        let emitter = EventEmitter::<()>::new();
        let count = Arc::new(AtomicU32::new(0));
        let c = Arc::clone(&count);

        emitter.on("ev", move |_| {
            c.fetch_add(1, Ordering::SeqCst);
        });

        let cloned = emitter.clone();
        cloned.emit("ev", &());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
}
