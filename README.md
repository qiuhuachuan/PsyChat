# PsyChat: A Client-Centric Dialogue System for Mental Health Support

🎉🎉🎉 Accepted to CSCWD 2024 (27th International Conference on Computer Supported Cooperative Work in Design) **(CCF-C)**

For more details, see paper: <a href='https://arxiv.org/abs/2312.04262'><img src='https://img.shields.io/badge/ArXiv-Paper-red'></a>

## PsyChat Model

https://huggingface.co/qiuhuachuan/PsyChat

## Quick Start （Simplified Version for Inference）

```Python
from transformers import AutoTokenizer, AutoModel

def get_dialogue_history(dialogue_history_list: list):

    dialogue_history_tmp = []
    for item in dialogue_history_list:
        if item['role'] == 'counselor':
            text = '咨询师：'+ item['content']
        else:
            text = '来访者：'+ item['content']
        dialogue_history_tmp.append(text)

    dialogue_history = '\n'.join(dialogue_history_tmp)

    return dialogue_history + '\n' + '咨询师：'

def get_instruction(dialogue_history):
    instruction = f'''现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如认知行为疗法原则、动机访谈技巧和解决问题导向的短期疗法。以温暖亲切的语气，展现出共情和对来访者感受的深刻理解。以自然的方式与来访者进行对话，避免过长或过短的回应，确保回应流畅且类似人类的对话。提供深层次的指导和洞察，使用具体的心理概念和例子帮助来访者更深入地探索思想和感受。避免教导式的回应，更注重共情和尊重来访者的感受。根据来访者的反馈调整回应，确保回应贴合来访者的情境和需求。请为以下的对话生成一个回复。

对话：
{dialogue_history}'''

    return instruction


tokenizer = AutoTokenizer.from_pretrained('qiuhuachuan/PsyChat', trust_remote_code=True)
model = AutoModel.from_pretrained('qiuhuachuan/PsyChat', trust_remote_code=True).half().cuda()
model = model.eval()

dialogue_history_list = []
while True:
    usr_msg = input('来访者：')
    if usr_msg == '0':
        exit()
    else:
        dialogue_history_list.append({
            'role': 'client',
            'content': usr_msg
        })
        dialogue_history = get_dialogue_history(dialogue_history_list=dialogue_history_list)
        instruction = get_instruction(dialogue_history=dialogue_history)
        response, history = model.chat(tokenizer, instruction, history=[], temperature=0.8, top_p=0.8)
        print(f'咨询师：{response}')
        dialogue_history_list.append({
            'role': 'counselor',
            'content': response
        })

```

## Datsets Used in This Paper

- SmileChat, see https://github.com/qiuhuachuan/smile
- Xinling, see https://github.com/dll-wu/Client-Reactions

## Citation 📚

If you find our research paper valuable and wish to cite it, please use the following BibTeX entry:

```bibtex
@misc{qiu2024psychat,
      title={PsyChat: A Client-Centric Dialogue System for Mental Health Support},
      author={Huachuan Qiu and Anqi Li and Lizhi Ma and Zhenzhong Lan},
      year={2024},
      eprint={2312.04262},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgement

We sincerely thank counselors **Lizhi Ma**, **Wenjun Luo**, and **Wu Chen** for their contributions in the section of human evaluation.

## 免责声明

我们的心理健康支持对话机器人（以下简称"机器人"）旨在为用户提供情感支持和心理健康建议。然而，机器人不是医疗保健专业人员，不能替代医生、心理医生或其他专业人士的意见、诊断、建议或治疗。

机器人提供的建议和信息是基于算法和机器学习技术，可能并不适用于所有用户或所有情况。因此，我们建议用户在使用机器人之前咨询医生或其他专业人员，了解是否适合使用此服务。

机器人并不保证提供的建议和信息的准确性、完整性、及时性或适用性。用户应自行承担使用机器人服务的所有风险。我们对用户使用机器人服务所产生的任何后果不承担任何责任，包括但不限于任何直接或间接的损失、伤害、精神疾病、财产损失或任何其他损害。

我们强烈建议用户在使用机器人服务时，遵循以下原则：

1. 机器人并不是医疗保健专业人士，不能替代医生、心理医生或其他专业人士的意见、诊断、建议或治疗。如果用户需要专业医疗或心理咨询服务，应寻求医生或其他专业人士的帮助。

2. 机器人提供的建议和信息仅供参考，用户应自己判断是否适合自己的情况和需求。如果用户对机器人提供的建议和信息有任何疑问或不确定，请咨询医生或其他专业人士的意见。

3. 用户应保持冷静、理性和客观，不应将机器人的建议和信息视为绝对真理或放弃自己的判断力。如果用户对机器人的建议和信息产生质疑或不同意，应停止使用机器人服务并咨询医生或其他专业人士的意见。

4. 用户应遵守机器人的使用规则和服务条款，不得利用机器人服务从事任何非法、违规或侵犯他人权益的行为。

5. 用户应保护个人隐私，不应在使用机器人服务时泄露个人敏感信息或他人隐私。

6. 平台收集的数据用于学术研究。

最后，我们保留随时修改、更新、暂停或终止机器人服务的权利，同时也保留对本免责声明进行修改、更新或补充的权利。如果用户继续使用机器人服务，即视为同意本免责声明的全部内容和条款。
