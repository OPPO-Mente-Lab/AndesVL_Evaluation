from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes
from PIL import Image
import uuid
import hmac


def sign(params, body, app_id, secret_key):
    # 1. 构建认证字符串前缀，格式为 bot-auth-v1/{appId}/{timestamp}, timestamp为时间戳，精确到毫秒，用以验证请求是否失效
    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    sb = [auth_string_prefix]
    # 2. 构建url参数字符串，按照参数名字典序升序排列
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        sb.extend(["{}={}&".format(k, v) for k, v in ordered_params.items()])
    # 3. 拼接签名原文字符串
    sign_str = "".join(sb) + body
    # 4. hmac_sha_256算法签名
    signature = hmac.new(secret_key.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha256).hexdigest()
    # 5. 拼接认证字符串
    return auth_string_prefix + signature

url = "https://https://andesgpt-gateway.oppoer.me/chat/v1/completions"
ak='RongZhiLab'
sk='3qeBDOdPLgrOUPU7NE3IAMdOxQ6E1Ksh4oMmqMPwhTI='


class Claude_Wrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'claude-3-opus-20240229',
                 headers = None,
                 key: str = None,
                 retry: int = 10,
                 wait: int = 3,
                 system_prompt: str = None,
                 verbose: bool = True,
                 temperature: float = 0,
                 max_tokens: int = 4096,
                 img_size: int=512,
                 img_detail: str = 'high',
                 **kwargs):

        self.model = model
        self.img_size = img_size
        self.img_detail=img_detail
        self.headers = headers
        self.temperature = temperature
        self.max_tokens = max_tokens

        super().__init__(retry=retry, wait=wait, verbose=verbose, system_prompt=system_prompt, **kwargs)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text' and msg['value'] != '':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    pth = msg['value']
                    suffix = osp.splitext(pth)[-1].lower()
                    media_type = mimetypes.types_map.get(suffix, None)
                    assert media_type is not None

                    content_list.append(dict(
                        type='image',
                        source={
                            'type': 'base64',
                            'media_type': media_type,
                            'data': encode_image_file_to_base64(pth, target_size=4096)
                        }))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    # def prepare_inputs(self, inputs):
    #     input_msgs = []
    #     assert isinstance(inputs, list) and isinstance(inputs[0], dict)
    #     assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
    #     if 'role' in inputs[0]:
    #         assert inputs[-1]['role'] == 'user', inputs[-1]
    #         for item in inputs:
    #             input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
    #     else:
    #         input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
    #     return input_msgs

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content = ""
            images = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content = msg['value']
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
                    images.append(img_struct)
            input_msgs.append(dict(role='user', content=content, images=images))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            input_msgs.append(dict(role='user', content=text))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:

        payload = json.dumps({
            'model': self.model,
            'max_tokens': self.max_tokens,
            'messages': self.prepare_inputs(inputs),
            'system': self.system_prompt,
            **kwargs
        })
        headers = {
            "recordId": str(uuid.uuid1()),
            "Authorization": sign(None, payload, ak, sk),
            "Content-Type": "application/json"
        }
        response = requests.request('POST', url, headers=headers, data=payload)
        if json.loads(response.content)["code"] == -20001:
            print("retry!")
            time.sleep(60)
            response = requests.request(
                "POST",
                url=url,
                headers=headers,
                data=payload,
            )
        
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg

        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['data']['choices'][0]['message']['content']
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(response.text if hasattr(response, 'text') else response)

        return ret_code, answer, response


class Claude3V(Claude_Wrapper):

    def generate(self, message, dataset=None):
        return super(Claude_Wrapper, self).generate(message)
