import time
import asyncio
import os
import tempfile
import numpy as np
import librosa
import edge_tts
from io import BytesIO

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "edgetts")
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg
        voice = self.opt.REF_FILE or "zh-CN-YunxiaNeural"
        voicename = textevent.get('tts', {}).get('ref_file',voice) #self.opt.REF_FILE #"zh-CN-YunxiaNeural"
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        logger.info(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            logger.error('edgetts err!!!!!')
            return

        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint={}
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text}
            eventpoint.update(**textevent) #eventpoint={'status':'end','text':text,'msgevent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate()

    def __create_bytes_stream(self,byte_stream):
        # edge_tts streams MP3; write to a temp file so librosa can fall back
        # to audioread/ffmpeg when the installed soundfile does not support MP3.
        suffix = '.mp3'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(byte_stream.read())
            tmp_path = tmp.name
        try:
            stream, sample_rate = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
            logger.info(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
            stream = stream.astype(np.float32)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            logger.exception('edgetts')
