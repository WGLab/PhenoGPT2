from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import spacy, torch
# define label maps
id2label = {0: "NOISE", 1: "INFORMATIVE"}
label2id = {"NOISE":0, "INFORMATIVE":1}
chunk_nlp = spacy.load("en_core_web_sm")
def chunking_documents(query, bert_tokenizer, bert_model, word_count=300, word_overlap=50):
    assert word_overlap < word_count, "word_overlap must be smaller than word_count"

    # Clean up extra whitespace
    query_split = query.split()
    query = ' '.join([q.strip() for q in query_split if len(q.strip()) > 0])

    # Optional: run spaCy to normalize spacing/sentences, but not needed for word chunking
    doc = chunk_nlp(query)
    words = [token.text for token in doc if not token.is_space]

    chunks = []
    start = 0
    while start < len(words):
        end = start + word_count
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        if end >= len(words):
            break
        start += word_count - word_overlap

    return chunks
def tokenize_function(bert_tokenizer, examples):
    # extract text
    text = examples["text"]
    #tokenize and truncate text
    bert_tokenizer.truncation_side = "left"
    tokenized_inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding='max_length',
        max_length=512
    )
    return tokenized_inputs
def predict_label(bert_tokenizer, bert_model, data_point):
    inputs = tokenize_function(bert_tokenizer, data_point)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return id2label[predicted_class_id]

def bert_init(local_dir = '/home/nguyenqm/projects/M2D2AI/scripts/note_filtering/BioLinkBERT_300Filtering_BalancedLabels_2025-05-13 01:34:58.307855/model/'):
    bert_tokenizer = AutoTokenizer.from_pretrained(local_dir)


    # generate classification bert_model from model_checkpoint
    bert_model = AutoModelForSequenceClassification.from_pretrained(
        local_dir, num_labels=2, id2label=id2label, label2id=label2id)
    # add pad token if none exists
    if bert_tokenizer.pad_token is None:
        bert_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        bert_model.resize_token_embeddings(len(bert_tokenizer))
    return bert_tokenizer, bert_model
