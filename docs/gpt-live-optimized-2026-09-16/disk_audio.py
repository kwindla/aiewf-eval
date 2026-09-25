"""Incremental copies of paced audio tracks; raw PCM survives process termination."""
import json,time,wave
import numpy as np
from pipecat.frames.frames import InputAudioRawFrame,OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor,FrameDirection
class DiskAudio(FrameProcessor):
 def __init__(self,path):
  super().__init__();self.path=path;self.count={'user':0,'assistant':0}
  self.files={k:(path/(k+'-track.pcm')).open('wb',buffering=0) for k in self.count}
  self.mapping=(path/'playback-samples.jsonl').open('w',buffering=1)
 async def process_frame(self,frame,direction):
  await super().process_frame(frame,direction)
  if direction==FrameDirection.DOWNSTREAM and isinstance(frame,(InputAudioRawFrame,OutputAudioRawFrame)):
   if frame.sample_rate!=24000 or frame.num_channels!=1:raise ValueError('Expected 24kHz mono recording frames')
   k='user' if isinstance(frame,InputAudioRawFrame) else 'assistant';start=self.count[k];self.files[k].write(frame.audio);self.count[k]+=len(frame.audio)//2
   self.mapping.write(json.dumps(dict(monotonic=time.monotonic(),track=k,start_sample=start,end_sample=self.count[k]))+'\n')
  await self.push_frame(frame,direction)
 def close_tracks(self):
  for f in self.files.values():f.close()
  self.mapping.close()
  arrays=[np.fromfile(self.path/(k+'-track.pcm'),dtype='<i2') for k in ['user','assistant']]
  stereo=np.zeros((max(map(len,arrays)),2),dtype='<i2')
  for i,a in enumerate(arrays):stereo[:len(a),i]=a
  with wave.open(str(self.path/'incremental-conversation.wav'),'wb') as f:
   f.setnchannels(2);f.setsampwidth(2);f.setframerate(24000);f.writeframes(stereo.tobytes())
