from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.tf_idf_embeddings import TfIdfGenerator
from modules.topic_model import TopicModel
from modules.evaluation import Evaluation



CONSTANTS = {
    "dataset":"UN_100w",
    "embedding_method":"tf-idf",
    "model":"KM",
    "topic_method":"tfidf"
}

def run_experiment():
    print("Reading Data")
    dr = DataReader()#filename="scientific_parallel.csv")
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en)
    prepped_text = tp.prepare_text()

    spanish_tp = TextPreparation(text_data.es,language="spanish")
    spanish_prepped_text = spanish_tp.prepare_text(pipeline=["clean","lemmatize","filter"])
    spanish_prepped_text = spanish_prepped_text.loc[tp.indexes_to_keep]

    full_text = prepped_text.append(spanish_prepped_text)

    print("Calculating Embeddings")
    teg = TfIdfGenerator(full_text)
    teg.fit_tf_idf()
    
    embeddings = teg.calculate_embeddings(prepped_text)

    print("Generating Topics")
    model = TopicModel(model_name="KM")
    topics = model.get_topics(embeddings,save=True)

    del embeddings

    spanish_embeddings = teg.calculate_embeddings(spanish_prepped_text)
    del teg
    spanish_topics = model.get_topics(spanish_embeddings,save=False)
    del spanish_embeddings

    print("Calculating Utilities")
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens(topics,method="tfidf")
    coherence = utils.get_coherence(top_tokens)

    spanish_utils = Evaluation()
    spanish_utils.create_utility_objects(spanish_prepped_text)
    spanish_top_tokens = spanish_utils.get_top_topic_tokens(spanish_topics,method="tfidf")
    spanish_coherence = spanish_utils.get_coherence(spanish_top_tokens)


    diversity= utils.get_topic_diversity(top_tokens)
    spanish_diversity= spanish_utils.get_topic_diversity(spanish_top_tokens)
    

    average_matching = utils.average_topic_matching(topics,spanish_topics)

    cla = utils.get_cross_lingual_alignment(topics,spanish_topics,prepped_text.values,spanish_prepped_text.values)

    other_stats = utils.get_dataset_stats(prepped_text)


    final_object = {"coherence":coherence,"top_tokens":top_tokens,"spanish_coherence":spanish_coherence,"spanish_top_tokens":spanish_top_tokens,
                    "diversity":diversity,"spanish_diversity":spanish_diversity,"average_matching":average_matching,
                    "cla":cla,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_tf_idf_KM_tfidftopic.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




