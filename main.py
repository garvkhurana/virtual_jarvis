import threading
import time
import queue
import cv2 as cv
from dataclasses import dataclass
from typing import Optional, Dict, Any
import signal
import sys
from object_detection import initialize_model, classify_frame, draw_prediction
from llm_response import LLMResponse
from tts import TTS
from user_query import speech_to_text

@dataclass
class PipelineConfig:
    detection_interval: float = 0.5 
    query_timeout: float = 10.0      
    interrupt_check_interval: float = 0.1 
    max_queue_size: int = 10        

class CandiceMainPipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()):
        self.config = config
        
        self.llm_response = LLMResponse(query="", detected_object="")
        self.tts = TTS()
        self.stt = speech_to_text()
        
        self.detection_queue = queue.Queue(maxsize=config.max_queue_size)
        self.query_queue = queue.Queue(maxsize=config.max_queue_size)
        self.response_queue = queue.Queue(maxsize=config.max_queue_size)
        
        self.stop_event = threading.Event()
        self.interrupt_event = threading.Event()
        self.listening_event = threading.Event()
        self.speaking_event = threading.Event()
        
        self.current_detection = None
        self.current_query = None
        self.is_processing = False
        
        self.detection_thread = None
        self.query_thread = None
        self.response_thread = None
        self.main_thread = None
        
        self.cap = None
        
    def initialize(self) -> bool:
        try:
            print("🔧 Initializing Candice Pipeline...")
            
            if not initialize_model():
                print(" Failed to initialize YOLO models")
                return False
            
            self.cap = cv.VideoCapture(0)
            if not self.cap.isOpened():
                print(" Failed to initialize camera")
                return False
                
            print(" All components initialized successfully!")
            return True
            
        except Exception as e:
            print(f" Initialization error: {e}")
            return False
    
    def object_detection_worker(self):
        print(" Object detection thread started")
        
        while not self.stop_event.is_set():
            try:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv.flip(frame, 1)
                prediction = classify_frame(frame)
                
                if prediction:
                    self.current_detection = prediction
                    
                    try:
                        self.detection_queue.put_nowait(prediction)
                    except queue.Full:
                        try:
                            self.detection_queue.get_nowait()
                            self.detection_queue.put_nowait(prediction)
                        except queue.Empty:
                            pass
                
                frame = draw_prediction(frame, prediction)
                
                status_y = 30
                if self.listening_event.is_set():
                    cv.putText(frame, "🎤 LISTENING...", (10, status_y), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    status_y += 30
                
                if self.speaking_event.is_set():
                    cv.putText(frame, " SPEAKING...", (10, status_y), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    status_y += 30
                
                if self.is_processing:
                    cv.putText(frame, " THINKING...", (10, status_y), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv.putText(frame, "Press Q to quit | Press SPACE for voice query", 
                          (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv.imshow('Candice - AI Assistant', frame)
                
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop_event.set()
                    break
                elif key == ord(' ') and not self.listening_event.is_set() and not self.speaking_event.is_set():
                    self.trigger_voice_query()
                
                time.sleep(self.config.detection_interval)
                
            except Exception as e:
                print(f" Detection thread error: {e}")
                continue
        
        print(" Object detection thread stopped")
    
    def query_worker(self):
        print(" Query thread started")
        
        while not self.stop_event.is_set():
            try:
                if not self.listening_event.wait(timeout=1.0):
                    continue
                
                print(" Listening for your query...")
                
                try:
                    query_text = self.stt.record_and_transcribe()
                    
                    if query_text and query_text.strip():
                        print(f" Query received: '{query_text}'")
                        
                        # Put query in queue
                        query_data = {
                            'text': query_text.strip(),
                            'timestamp': time.time(),
                            'detection': self.current_detection
                        }
                        
                        try:
                            self.query_queue.put_nowait(query_data)
                        except queue.Full:
                            try:
                                self.query_queue.get_nowait()
                                self.query_queue.put_nowait(query_data)
                            except queue.Empty:
                                pass
                    else:
                        print(" No valid query detected")
                        
                except Exception as e:
                    print(f" Query recording error: {e}")
                
                finally:
                    self.listening_event.clear()
                
            except Exception as e:
                print(f" Query thread error: {e}")
                continue
        
        print(" Query thread stopped")
    
    def response_worker(self):
        print(" Response thread started")
        
        while not self.stop_event.is_set():
            try:
                try:
                    query_data = self.query_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self.is_processing = True
                print(" Processing query with LLM...")
                
                query_text = query_data['text']
                detection = query_data.get('detection')
                detected_object = detection['object_name'] if detection else "None"
                
                try:
                    llm_response = self.llm_response.get_response(
                        query=query_text,
                        detected_object=detected_object
                    )
                    
                    print(f" LLM Response: {llm_response}")
                    
                    if not self.interrupt_event.is_set():
                        self.speaking_event.set()
                        print(" Speaking response...")
                        
                        self.tts.generate_tts(llm_response)
                        
                        self.speaking_event.clear()
                    else:
                        print(" Response interrupted")
                        self.interrupt_event.clear()
                
                except Exception as e:
                    print(f" LLM processing error: {e}")
                
                finally:
                    self.is_processing = False
                
            except Exception as e:
                print(f" Response thread error: {e}")
                continue
        
        print(" Response thread stopped")
    
    def trigger_voice_query(self):
        if self.speaking_event.is_set():
            self.interrupt_event.set()
            self.speaking_event.clear()
            print(" Interrupting current response...")
            time.sleep(0.5)  
        
        while not self.query_queue.empty():
            try:
                self.query_queue.get_nowait()
            except queue.Empty:
                break
        
        self.listening_event.set()
    
    def start_threads(self):
        print(" Starting worker threads...")
        
        self.detection_thread = threading.Thread(
            target=self.object_detection_worker,
            name="ObjectDetection",
            daemon=True
        )
        
        self.query_thread = threading.Thread(
            target=self.query_worker,
            name="QueryHandler",
            daemon=True
        )
        
        self.response_thread = threading.Thread(
            target=self.response_worker,
            name="ResponseHandler",
            daemon=True
        )
        
        self.detection_thread.start()
        self.query_thread.start()
        self.response_thread.start()
        
        print(" All threads started successfully!")
    
    def stop_threads(self):
        print(" Stopping threads...")
        
        self.stop_event.set()
        self.interrupt_event.set()
        self.listening_event.clear()
        self.speaking_event.clear()
        
        threads = [self.detection_thread, self.query_thread, self.response_thread]
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        print("All threads stopped")
    
    def cleanup(self):
        print(" Cleaning up resources...")
        
        if self.cap:
            self.cap.release()
        
        cv.destroyAllWindows()
        print(" Cleanup complete")
    
    def run(self):
        try:
            if not self.initialize():
                return False
            
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.start_threads()
            
            print("\n" + "="*60)
            print(" CANDICE AI ASSISTANT IS READY!")
            print("="*60)
            print(" Instructions:")
            print("   • Press SPACE to ask a voice query")
            print("   • Press Q to quit")
            print("   • Say 'forget it' to clear memory")
            print("   • Interrupt anytime by pressing SPACE during speech")
            print("="*60 + "\n")
            
            # Keep main thread alive
            while not self.stop_event.is_set():
                time.sleep(0.1)
            
            return True
            
        except KeyboardInterrupt:
            print("\n Keyboard interrupt received")
        except Exception as e:
            print(f" Pipeline error: {e}")
        finally:
            self.stop_threads()
            self.cleanup()
            
        return False
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown"""
        print(f"\n Received signal {signum}, shutting down gracefully...")
        self.stop_event.set()

def main():
    print(" Starting Candice AI Assistant...")
    
    config = PipelineConfig(
        detection_interval=0.3, 
        query_timeout=10.0,
        interrupt_check_interval=0.1,
        max_queue_size=5
    )
    
    pipeline = CandiceMainPipeline(config)
    success = pipeline.run()
    
    if success:
        print(" Candice AI Assistant terminated successfully!")
    else:
        print(" Candice AI Assistant terminated with errors!")
    
    return 0 if success else 1

