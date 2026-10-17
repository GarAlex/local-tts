//! Keep device creation, playback and device teardown off the egui thread.
use local_tts_core::Audio;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
    mpsc,
};

pub struct Player {
    tx: mpsc::SyncSender<()>,
    latest: Arc<Mutex<Option<(u64, Audio)>>>,
    current: Arc<AtomicU64>,
    pending: Option<u64>,
}
trait Backend {
    fn play(&mut self, audio: Audio, current: &AtomicU64, id: u64) -> Result<(), String>;
    fn stop(&mut self);
}
#[derive(Default)]
struct Device {
    active: Option<(rodio::OutputStream, rodio::Sink)>,
}
impl Backend for Device {
    fn play(&mut self, audio: Audio, current: &AtomicU64, id: u64) -> Result<(), String> {
        self.stop();
        let (stream, handle) = rodio::OutputStream::try_default().map_err(|e| e.to_string())?;
        // A slow driver may return after Stop or a newer request. Never start that audio.
        if current.load(Ordering::SeqCst) != id {
            return Ok(());
        }
        let sink = rodio::Sink::try_new(&handle).map_err(|e| e.to_string())?;
        if current.load(Ordering::SeqCst) != id {
            return Ok(());
        }
        sink.append(rodio::buffer::SamplesBuffer::new(
            1,
            audio.sample_rate,
            audio.samples,
        ));
        self.active = Some((stream, sink));
        Ok(())
    }
    fn stop(&mut self) {
        self.active = None;
    }
}
impl Player {
    pub fn new(report: impl Fn(u64, Result<(), String>) + Send + 'static) -> Self {
        Self::worker(Device::default, report)
    }
    fn worker<B: Backend + 'static>(
        factory: impl FnOnce() -> B + Send + 'static,
        report: impl Fn(u64, Result<(), String>) + Send + 'static,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel(1);
        let latest = Arc::new(Mutex::new(None::<(u64, Audio)>));
        let worker_latest = latest.clone();
        let current = Arc::new(AtomicU64::new(0));
        let worker_current = current.clone();
        std::thread::spawn(move || {
            // Backend is constructed here; non-Send audio resources never cross threads.
            let mut device = factory();
            while rx.recv().is_ok() {
                let request = worker_latest.lock().unwrap().take();
                if let Some((id, audio)) = request {
                    if worker_current.load(Ordering::SeqCst) == id {
                        let result = device.play(audio, &worker_current, id);
                        report(id, result);
                    }
                } else {
                    device.stop();
                }
            }
        });
        Self {
            tx,
            latest,
            current,
            pending: None,
        }
    }
    pub fn play(&mut self, audio: Audio) {
        let id = self.current.fetch_add(1, Ordering::SeqCst) + 1;
        self.pending = Some(id);
        // Retain at most one queued clip while a device driver is stalled.
        *self.latest.lock().unwrap() = Some((id, audio));
        let _ = self.tx.try_send(());
    }
    pub fn stop(&mut self) {
        self.current.fetch_add(1, Ordering::SeqCst);
        self.pending = None;
        *self.latest.lock().unwrap() = None;
        let _ = self.tx.try_send(());
    }
    pub fn pending(&self) -> bool {
        self.pending.is_some()
    }
    pub fn accept(&mut self, id: u64) -> bool {
        if self.pending == Some(id) {
            self.pending = None;
            true
        } else {
            false
        }
    }
}
impl Drop for Player {
    fn drop(&mut self) {
        // Never join: an OS audio driver can block indefinitely during initialization.
        self.stop();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    struct Stalled {
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        started: mpsc::Sender<u64>,
    }
    impl Backend for Stalled {
        fn play(&mut self, _: Audio, current: &AtomicU64, id: u64) -> Result<(), String> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            if current.load(Ordering::SeqCst) == id {
                self.started.send(id).unwrap();
            }
            Err("test device unavailable".into())
        }
        fn stop(&mut self) {}
    }
    #[test]
    fn blocked_driver_does_not_block_stop_or_drop_and_cannot_start_stale_audio() {
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let (started_tx, started) = mpsc::channel();
        let (reported_tx, reported) = mpsc::channel();
        let mut player = Player::worker(
            move || Stalled {
                entered: entered_tx,
                release: release_rx,
                started: started_tx,
            },
            move |id, result| {
                let _ = reported_tx.send((id, result));
            },
        );
        let audio = Audio {
            sample_rate: 22050,
            samples: vec![0.0; 10],
        };
        player.play(audio.clone());
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        player.stop();
        for _ in 0..100 {
            player.play(audio.clone());
        }
        let newest = player.current.load(Ordering::SeqCst);
        assert_eq!(player.latest.lock().unwrap().as_ref().unwrap().0, newest);
        release.send(()).unwrap();
        let (old_id, _) = reported.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(!player.accept(old_id));
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        release.send(()).unwrap();
        let (id, result) = reported.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(player.accept(id));
        assert_eq!(result.unwrap_err(), "test device unavailable");
        assert_eq!(started.recv_timeout(Duration::from_secs(2)).unwrap(), id);
        assert!(started.try_recv().is_err());
        // Drop while initialization is blocked also must return without waiting.
        player.play(Audio {
            sample_rate: 22050,
            samples: vec![0.0],
        });
        entered.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(player);
        release.send(()).unwrap();
        let _ = reported.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(started.try_recv().is_err());
    }
}
