from transformers import pipeline
import random
# from tool.csv_dataaugment import unmasker
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoConfig
from tools.config_loader import get_config
import os


class Text_Augmentation(object):
    def __init__(self, config):
        """
        method: instead, insert, translation
        """

        self.method = config.Text_Augment.method

        if self.method == 'instead':
            self.pipline = pipeline('fill-mask', model='bert-base-cased')

        elif self.method == 'insert':
            self.pipline = pipeline('fill-mask', model='bert-base-cased')

        elif self.method == 'translation':
            self.translator_en_to_de = AutoConfig.from_pretrained('t5-base')
            self.tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>", bos_token="<s>")
            self.model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
              #self.pipline = pipeline('translation_en_to_de', model='t5-base')

    def batch_generate(self, caption_list):
        """
        caption_list: ['i am zjz', 'i am a pig']
        """
        augmented_text_list = [self.generate(self.method, caption) for caption in caption_list]      #using the generate function
        return augmented_text_list

    def generate(self, method, caption):
        """
        caption: str
        """
        if method == 'instead':
            input_text = caption  # "I went to see a movie in the theater"
            orig_text_list = input_text.split()
            len_input = len(orig_text_list)
            # Random index where we want to replace the word
            rand_idx = random.randint(1, len_input - 1)
            orig_word = orig_text_list[rand_idx]
            new_text_list = orig_text_list.copy()
            new_text_list[rand_idx] = '[MASK]'
            new_mask_sent = ' '.join(new_text_list)
            # print("Masked sentence->", new_mask_sent)
            # I went to [MASK] a movie in the theater
            augmented_text_list = self.pipline(new_mask_sent)
            # To ensure new word and old word are not name
            for res in augmented_text_list:
                if res['token_str'] != orig_word:
                    augmented_text = res['sequence']
                    break
            # print("Augmented text->", augmented_text)
            return augmented_text

        elif method == 'insert':
             input_text = caption

             orig_text_list = input_text.split()
             len_input = len(orig_text_list)

             # Random index where we want to insert the word except at the start or end
             rand_idx = random.randint(1, len_input - 2)

             new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx:]
             new_mask_sent = ' '.join(new_text_list)
             #print("Masked sentence->", new_mask_sent)
             # I went to see a [Mask] movie in the theater
             augmented_text_list = self.pipline(new_mask_sent)
             augmented_text = augmented_text_list[0]['sequence']
             print("Augmented text->", augmented_text)
             return augmented_text

        elif method == 'translation':
             input_text = caption
             en_to_de_output = self.translator_en_to_de(input_text)
             translated_text = en_to_de_output[0]['translation_text']
             print("Translated text->", translated_text)
             input_ids = self.tokenizer(translated_text, return_tensors="pt", add_special_tokens=False).input_ids
             output_ids = self.model_de_to_en.generate(input_ids)[0]
             augmented_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
             print("Augmented Text->", augmented_text)
             return augmented_text










#text_augmentation = Text_Augmentation(method='instead')
#print('hh')
#pass


#def unmasker_instead(itemcaptions):
    #"""

    #"""

    #unmasker = pipeline('fill-mask', model='bert-base-cased')
    #input_text = itemcaptions  # "I went to see a movie in the theater"
    #orig_text_list = input_text.split()
    #len_input = len(orig_text_list)
    # Random index where we want to replace the word
    #rand_idx = random.randint(1, len_input - 1)
    #orig_word = orig_text_list[rand_idx]
    #new_text_list = orig_text_list.copy()
    #new_text_list[rand_idx] = '[MASK]'
    #new_mask_sent = ' '.join(new_text_list)
    #print("Masked sentence->", new_mask_sent)
    # I went to [MASK] a movie in the theater
    #augmented_text_list = unmasker(new_mask_sent)
    # To ensure new word and old word are not name
    #for res in augmented_text_list:
        #if res['token_str'] != orig_word:
            #augmented_text = res['sequence']
            #break
    #print("Augmented text->", augmented_text)
    # I went to watch a movie in the theater
    #return augmented_text


#def length(itemcaptions):
   # generator = pipeline('text-generation', model='gpt2')
   # input_text = itemcaptions  # "I went to see a movie in the theater"
   # input_length = len(input_text.split())
   # num_new_words = 5
   # output_length = input_length + num_new_words
   # gpt_output = generator(input_text, max_length=output_length, num_return_sequences=5)
   # augmented_text = gpt_output[0]['generated_text']
   # print("Augmented text->", augmented_text)
    # I went to see a movie in the theater, and the director was
    # return augmented_text


#def unmasker_insert(itemcaptions):
    # os.environ['HF_TRANSFORMERS_CACHE'] = '/vol/research/Audiocapt/NLP/cache/'

    #unmasker = pipeline('fill-mask', model='bert-base-cased')

    #input_text = itemcaptions

    #orig_text_list = input_text.split()
    #len_input = len(orig_text_list)

    # Random index where we want to insert the word except at the start or end
    #rand_idx = random.randint(1, len_input - 2)

    #new_text_list = orig_text_list[:rand_idx] + ['[MASK]'] + orig_text_list[rand_idx:]
    #new_mask_sent = ' '.join(new_text_list)
    #print("Masked sentence->", new_mask_sent)
    # I went to see a [Mask] movie in the theater

    #augmented_text_list = unmasker(new_mask_sent)
    #augmented_text = augmented_text_list[0]['sequence']
    #print("Augmented text->", augmented_text)
    #return augmented_text
    # I went to see a new movie in the theater


#def translation(itemcaptions):
    # translator_en_to_de = pipeline('translation_en_to_de', model='t5-base')
    #translator_en_to_de = AutoConfig.from_pretrained('t5-base')
    #tokenizer = AutoTokenizer.from_pretrained("google/bert2bert_L-24_wmt_de_en", pad_token="<pad>", eos_token="</s>",
                                              #bos_token="<s>")
    #model_de_to_en = AutoModelForSeq2SeqLM.from_pretrained("google/bert2bert_L-24_wmt_de_en")
    #input_text = itemcaptions
    #en_to_de_output = translator_en_to_de(input_text)
    #translated_text = en_to_de_output[0]['translation_text']
    #print("Translated text->", translated_text)
    #input_ids = tokenizer(translated_text, return_tensors="pt", add_special_tokens=False).input_ids
    #output_ids = model_de_to_en.generate(input_ids)[0]
    #augmented_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    #print("Augmented Text->", augmented_text)
