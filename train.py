import utils
import tensorflow as tf
import pyarrow as pa
from datasets import Dataset
dataset = utils.convert_csv_to_squad_type_dataset("test_data.csv")
tokenized_data, tokenizer = utils.preprocess_data(dataset.iloc[:20000],"distilbert-base-uncased")
tokenized_data=Dataset.from_dict(tokenized_data)

del dataset
from transformers import DefaultDataCollator
data_collator = DefaultDataCollator(return_tensors="tf")
from transformers import create_optimizer

batch_size = 16
num_epochs = 2
total_train_steps = (len(tokenized_data) // batch_size) * num_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)
from transformers import TFAutoModelForQuestionAnswering
model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
tf_train_set = model.prepare_tf_dataset(
    tokenized_data,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)


model.compile(optimizer=optimizer)

model.fit(x=tf_train_set, epochs=3,validation_data = test_set)

model.save_pretrained("Address-Split-Bert/Address-QA-model")
tokenizer.save_pretrained("Address-Split-Bert/Address-QA-Tokenizer")
from transformers import pipeline

street = "What is the Street?"
city = "What is the City?"
state = "What is the State?"
zip = "What is the Zip Code?"

context = "I reside in  256 Bark Avenue St. Petersburg CA 19312"
question_answerer = pipeline("question-answering", model=model,tokenizer=tokenizer)
print("Street:",question_answerer(question=street, context=context))
print("City:",question_answerer(question=city, context=context))
print("State:",question_answerer(question=state, context=context))
print("Zip Code:",question_answerer(question=zip, context=context))