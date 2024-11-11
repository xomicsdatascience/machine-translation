from datasets import load_dataset
from transformers import AutoTokenizer

translation_dataset = load_dataset('wmt14', 'de-en')
print(translation_dataset['train'][0])
print(len(translation_dataset['train']))

de_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

sample_de = [translation_dataset['train'][0]['translation']['de'], translation_dataset['train'][1]['translation']['de'], translation_dataset['train'][2]['translation']['de']]
sample_en = [translation_dataset['train'][0]['translation']['en'], translation_dataset['train'][1]['translation']['en'], translation_dataset['train'][2]['translation']['en']]

de_inputs = de_tokenizer(sample_de, return_tensors='pt', padding=True)
en_inputs = en_tokenizer(sample_en, return_tensors='pt', padding=True)

print("German Tokenization:")

print(sample_de)
print(f'unk token (german): {de_tokenizer.unk_token} ({de_tokenizer.convert_tokens_to_ids([de_tokenizer.unk_token])})')
print(f'sep token (german): {de_tokenizer.sep_token} ({de_tokenizer.convert_tokens_to_ids([de_tokenizer.sep_token])})')
print(f'pad token (german): {de_tokenizer.pad_token} ({de_tokenizer.convert_tokens_to_ids([de_tokenizer.pad_token])})')
print(f'cls token (german): {de_tokenizer.cls_token} ({de_tokenizer.convert_tokens_to_ids([de_tokenizer.cls_token])})')
print(f'mask token (german): {de_tokenizer.mask_token} ({de_tokenizer.convert_tokens_to_ids([de_tokenizer.mask_token])})')
print(de_inputs)

print("\nEnglish Tokenization:")
print(sample_en)
print(f'unk token (english): {en_tokenizer.unk_token} ({en_tokenizer.convert_tokens_to_ids([en_tokenizer.unk_token])})')
print(f'sep token (english): {en_tokenizer.sep_token} ({en_tokenizer.convert_tokens_to_ids([en_tokenizer.sep_token])})')
print(f'pad token (english): {en_tokenizer.pad_token} ({en_tokenizer.convert_tokens_to_ids([en_tokenizer.pad_token])})')
print(f'cls token (english): {en_tokenizer.cls_token} ({en_tokenizer.convert_tokens_to_ids([en_tokenizer.cls_token])})')
print(f'mask token (english): {en_tokenizer.mask_token} ({en_tokenizer.convert_tokens_to_ids([en_tokenizer.mask_token])})')
print(en_inputs)