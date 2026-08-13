import time
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar
from utils.logger import logger

def llm_response(message,avatar_session:'BaseAvatar',datainfo:dict={}):
    try:
        opt = avatar_session.opt
        start = time.perf_counter()
        from openai import OpenAI
        api_key = opt.llm_api_key or os.getenv('LLM_API_KEY') or os.getenv('DASHSCOPE_API_KEY') or 'not-needed'
        client = OpenAI(
            api_key=api_key,
            base_url=opt.llm_base_url,
        )
        end = time.perf_counter()
        logger.info(f"llm Time init: {end-start}s,{message}")
        completion = client.chat.completions.create(
            model=opt.llm_model,
            messages=[{'role': 'system', 'content': opt.llm_system_prompt},
                    {'role': 'user', 'content': message}],
            stream=True,
            stream_options={"include_usage": True}
        )
        result=""
        first = True
        for chunk in completion:
            if len(chunk.choices)>0:
                if first:
                    end = time.perf_counter()
                    logger.info(f"llm Time to first chunk: {end-start}s")
                    first = False
                msg = chunk.choices[0].delta.content
                if msg is None:
                    continue
                lastpos=0
                for i, char in enumerate(msg):
                    if char in ",.!;:，。！？：；" :
                        result = result+msg[lastpos:i+1]
                        lastpos = i+1
                        if len(result)>10:
                            logger.info(result)
                            avatar_session.put_msg_txt(result,datainfo)
                            result=""
                result = result+msg[lastpos:]
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end-start}s")
        if result:
            avatar_session.put_msg_txt(result,datainfo)

    except Exception as e:
        logger.exception('llm exception:')
        return   