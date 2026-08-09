#import pandas as pd
#from capstone.tokenizer import train_tokenizer, PAD_TOKEN, UNK_TOKEN

#def test_train_tokenizer():
#    texts = pd.Series([
#        "free moneys the click now!!",
#        "meeting notes attached",
#        "win the greatest prize today\x85",
#        "quarterly report review",
#    ])
#
#    tokenizer = train_tokenizer(texts, vocab_size=100, output_sequence_length=16)
#
#    assert tokenizer.token_to_id(PAD_TOKEN) is not None
#    assert tokenizer.token_to_id(UNK_TOKEN) is not None
#
#    encoded = tokenizer.encode("free money now")
#    assert len(encoded.ids) = 16

