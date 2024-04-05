import ujson

from datasets import Dataset, DatasetDict


def load_data(role:str, file_names: list):
    data_wrapper = {}
    for file_name in file_names:
        instances = {'context': [], 'response': [], 'label': []}
        with open(f'./data/{role}/{file_name}.json', 'r', encoding='utf-8') as f:
            data = ujson.load(f)

        for item in data:
            context = item['context']
            response = item['response']
            label = item['label']
            instances['context'].append(context)
            instances['response'].append(response)
            instances['label'].append(label)
        data_wrapper[file_name] = Dataset.from_dict(instances)

    dataset = DatasetDict(data_wrapper)
    return dataset


if __name__ == '__main__':
    '''
    To test, run this file in the root folder.
    `python ./utils/dataloader.py`
    '''
    dataset = load_data(role='client',file_names=['train', 'test'])
    print(dataset['test'][1])
