import streamlit as st
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

@st.cache_resource
def get_2_layer_model_and_tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    model = BartForConditionalGeneration.from_pretrained('./ckpt/trained_bart_braille_2_and_toknizer_init')
    model.model.encoder.layers = model.model.encoder.layers[:2]

    # 이진수 조합 토큰 추가
    binary_tokens = [
        '000000', '000001', '000010', '000011', '000100', '000101', '000110', 
        '000111', '001000', '001001', '001010', '001011', '001100', '001101', 
        '001110', '001111', '010000', '010001', '010010', '010011', '010100', 
        '010101', '010110', '010111', '011000', '011001', '011010', '011011', 
        '011100', '011101', '011110', '011111', '100000', '100001', '100010', 
        '100011', '100100', '100101', '100110', '100111', '101000', '101001', 
        '101010', '101011', '101100', '101101', '101110', '101111', '110000', 
        '110001', '110010', '110011', '110100', '110101', '110110', '110111', 
        '111000', '111001', '111010', '111011', '111100', '111101', '111110', 
        '111111'
    ]
    tokenizer.add_tokens(binary_tokens)

    return tokenizer, model


@st.cache_resource
def get_2_layer_model_and_tokenizer_2():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    model = BartForConditionalGeneration.from_pretrained('./ckpt/kobart_encoder_init')
    model.model.encoder.layers = model.model.encoder.layers[:2]

    return tokenizer, model


@st.cache_resource
def get_6_layer_model_and_tokenizer_best():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./ckpt/kobart_6_layer_and_13_epochs_re-tokenizer")
    model = BartForConditionalGeneration.from_pretrained('./ckpt/kobart_6_layer_and_13_epochs_re-tokenizer')
    return tokenizer, model



sample = '110001010010100100010110000001101110001101 101100000010101010100100101011 000100101100100000100010111110000110000010101110101010010000 101010110000100100101010010100010011110001010010100100010110000001101110001101'
st.title("점자 번역기")


option = st.selectbox(
    '모델 선택',
    ('trained_bart_braille_6_and_toknizer_init', 'trained_bart_braille_2_and_toknizer_init', 'kobart_encoder_init')
    #('trained_bart_braille_2_and_toknizer_init', 'kobart_encoder_init')
)

if option == 'trained_bart_braille_6_and_toknizer_init':
    tokenizer, model = get_6_layer_model_and_tokenizer_best()
elif option == 'trained_bart_braille_2_and_toknizer_init':
    tokenizer, model = get_2_layer_model_and_tokenizer()
elif option == 'kobart_encoder_init':
    tokenizer, model = get_2_layer_model_and_tokenizer_2()

user_input = st.text_area("Enter your binary braille:", sample, height=256)


if user_input is not None and user_input != "":
    tokenized_input = tokenizer.encode(user_input, return_tensors='pt')
    print(tokenized_input)
    generated_tokens = model.generate(
        input_ids=tokenized_input,
        # max_length=45,
        min_length=5,
        max_length=70,
        num_beams=2,
    )

    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(generated_text)
    st.write('# Result')
    st.write(generated_text)
