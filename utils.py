import numpy as np
import pandas as pd
import transformers


def generate_question_answer_pair(data_row, columns):
    questions = list(map(
                lambda column: "What is the {}?".format(column),
                columns))
    context = data_row["Address"]
    answers= [{"answer_start":[context.find(data_row[column])]
               ,"text":[data_row[column]]} for column in columns]
    return list(zip([context]*len(questions), questions, answers))


def convert_csv_to_squad_type_dataset(csv_file):
    """
    Takes input a csv file and returns a dataset in format of SQUAD dataset
    Converts Column-1 to Context and creates what is questions for the rest of the columns
    Example:
    Columns in CSV: [Address,Street,City]
        Context: Address
        Q1:What is the Street?
        Q2:What is the City?

    :param csv_file: file containing data
    :return: Tensorflow Dataset
    """
    initial_data = pd.read_csv(csv_file,dtype=str)
    initial_data.dropna(inplace=True)
    series_data = initial_data.apply(lambda data_row : generate_question_answer_pair(data_row, initial_data.columns[1:])
                                     , axis=1).explode('0')
    final_data = pd.DataFrame(np.array(series_data.apply(list).tolist())
                              , columns=["context", "question", "answers"])
    return final_data


def preprocess_data(dataframe, model):
    """
    Takes input a Pandas dataframe and preprocesses it for the specified model
    :param dataframe: PandasDataFrame: Dataframe containing data
    :param model: String :Name of the model
    :return: preprocessed data and tokenizer
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    questions = dataframe["question"].tolist()
    context = dataframe['context'].tolist()
    answers = dataframe['answers'].tolist()
    inputs = tokenizer(
        questions,
        context,
        max_length=384,
        truncation='only_second',
        return_offsets_mapping=True,
        padding='max_length',
    )

    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx-1

        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs,tokenizer



