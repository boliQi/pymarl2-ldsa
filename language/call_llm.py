import random
import time
from openai import OpenAI
import openai
import httpx

 
class LLM:
    def __init__(self, mode='openai') -> None:
        if mode == 'openai':
            api_key_list = [
                # 'f2841229-111e-482b-8753-96079f62ee6a',
                'tp-cujgv7fe7ka7m0m3azfp1p8wm5dnz3ekzfem4bnyvdp5yoz7'
            ]
            # self.agent_big = gpt_agent(random.choice(api_key_list), api_key_list, model_name='doubao-seed-1-8-251228')
            self.agent_big = gpt_agent(random.choice(api_key_list), api_key_list, model_name='mimo-v2.5-pro')
            self.agent_small = gpt_agent(random.choice(api_key_list), api_key_list, model_name='doubao-1-5-lite-32k-250115')
            self.call_llm = self.call_llm_openai
        else:
            assert 0
    
    def call_llm_openai(
        self,
        prompt,
        big_model=False,
        temperature=0.0,
        enable_thinking=False,
        thinking_budget_tokens=None,
    ):
        if big_model:
            return self.agent_big.ask(
                prompt,
                temperature=temperature,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
            )
        else:
            return self.agent_small.ask(
                prompt,
                temperature=temperature,
                enable_thinking=enable_thinking,
                thinking_budget_tokens=thinking_budget_tokens,
            )


class gpt_agent():

    def __init__(self, api_key: str, api_key_list, model_name="doubao-seed-1-6-251015") -> None:
        self.api_key = api_key
        self.ask_call_cnt = 0
        self.ask_call_cnt_sup = 3
        self.model_name = model_name
        self.api_key_list = api_key_list
        self._client = None

    def _get_client(self):
        """获取或创建 OpenAI client"""
        if self._client is None:
            # 开启连接复用 (Keep-Alive)，避免每次请求都显式握手，提升速度
            http_client = httpx.Client(
                timeout=httpx.Timeout(45.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            )
            self._client = OpenAI(
                api_key=self.api_key,
                # base_url="https://ark.cn-beijing.volces.com/api/v3",
                base_url = "https://token-plan-cn.xiaomimimo.com/v1",
                timeout=45.0,
                max_retries=2,
                http_client=http_client,
            )
        return self._client

    def _reset_client(self):
        """重置 client"""
        if self._client is not None:
            try:
                self._client.close()
            except:
                pass
            self._client = None

    def _extract_content(self, completion):
        """
        从 OpenAI SDK v1.x 的响应中提取内容
        """
        try:
            msg = completion.choices[0].message
        except Exception:
            msg = completion["choices"][0]["message"]

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict):
                    parts.append(p.get("text", "") or p.get("content", ""))
                else:
                    parts.append(str(p))
            content = "".join(parts)
        return content

    def ask(
        self,
        question,
        temperature=0.0,
        stop=None,
        enable_thinking=False,
        thinking_budget_tokens=None,
    ) -> str:
        res = "No answer!"
        self.ask_call_cnt += 1
        if self.ask_call_cnt > self.ask_call_cnt_sup:
            print("======> Achieve call count limit, Return!")
            self._random_key()
            self.ask_call_cnt = 0
            return res

        messages = [{"role": "user", "content": question}]
        thinking_cfg = {"type": "enabled" if enable_thinking else "disabled"}
        if enable_thinking and thinking_budget_tokens is not None:
            thinking_cfg["budget_tokens"] = int(thinking_budget_tokens)

        try:
            client = self._get_client()
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stop=stop,
                # Volcengine Ark (OpenAI-compatible) uses extra_body.thinking to control reasoning.
                extra_body={"thinking": thinking_cfg}
            )
            content = self._extract_content(completion)
            if content:
                res = content
            self.ask_call_cnt = 0
            
        except openai.AuthenticationError as e:
            self._random_key()
            print(f"======> AuthenticationError: {e}")
            
        except openai.RateLimitError as e:
            print(f"======> RateLimitError: {e}")
            self._random_key()
            time.sleep(10)
            return self.ask(question)
            
        except openai.APIConnectionError as e:
            print(f"======> APIConnectionError: {e}")
            self._reset_client()
            self._random_key()
            time.sleep(2)
            return self.ask(question)
            
        except openai.APITimeoutError as e:
            print(f"======> APITimeoutError: {e}")
            self._reset_client()
            return res
            
        except openai.APIStatusError as e:
            print(f"======> APIStatusError (status={e.status_code}): {e}")
            self._reset_client()
            if e.status_code >= 500:
                time.sleep(10)
                return self.ask(question)
                
        except Exception as e:
            print(f"======> Unexpected Exception: {type(e).__name__}: {e}")
            self._reset_client()
            self._random_key()
        
        return res

    def _random_key(self) -> None:
        self.api_key = random.choice(self.api_key_list)
        self._reset_client()