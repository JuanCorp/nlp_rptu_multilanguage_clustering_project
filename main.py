from modules.data_reader import DataReader
from modules.data_saver import DataSaver
from modules.text_prep import TextPreparation
from modules.text_embeddings import TextEmbeddingGenerator
from modules.topic_model import TopicModel
from modules.evaluation import Evaluation
import time

CONSTANTS = {
    "dataset":"scientific_papers",
    "embedding_method":"Pre-trained embeddings",
    "model":"GMM",
    "topic_method":"tf"
}


def run_experiment():
    print("Reading Data")
    dr = DataReader(filename="scientific_parallel.csv")
    text_data = dr.obtain_text_data()
    print(text_data.shape)
    
    print("Preparing Text")
    tp = TextPreparation(text_data.en)
    prepped_text = tp.prepare_text()

    start = time.time()
    print("Calculating Embeddings")
    teg = TextEmbeddingGenerator(prepped_text)#,model="paraphrase-multilingual-mpnet-base-v2")
    embeddings = teg.calculate_embeddings()
    embedding_model = teg.model
    teg.unload_transformer()
    del teg

    print("Generating Topics")
    model = TopicModel(model_name="GMM")
    topics = model.get_topics(embeddings,save=True)

    del embeddings
    end = time.time()
    training_time = end-start

    spanish_tp = TextPreparation(text_data.es,language="spanish")
    spanish_prepped_text = spanish_tp.prepare_text(pipeline=["clean","lemmatize","filter"])
    spanish_prepped_text = spanish_prepped_text.loc[tp.indexes_to_keep]
    teg = TextEmbeddingGenerator(spanish_prepped_text)#,model="paraphrase-multilingual-mpnet-base-v2")
    spanish_embeddings = teg.calculate_embeddings()
    teg.unload_transformer()
    del teg
    spanish_topics = model.get_topics(spanish_embeddings,save=False)
    del spanish_embeddings

    print("Calculating Utilities")
    print(topics)
    utils = Evaluation()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens(topics,method="freq")
    coherence = utils.get_coherence(top_tokens)

    spanish_utils = Evaluation()
    spanish_utils.create_utility_objects(spanish_prepped_text)
    spanish_top_tokens = spanish_utils.get_top_topic_tokens(spanish_topics,method="freq")
    spanish_coherence = spanish_utils.get_coherence(spanish_top_tokens)


    diversity= utils.get_topic_diversity(top_tokens)
    spanish_diversity= spanish_utils.get_topic_diversity(spanish_top_tokens)
    

    average_matching = utils.average_topic_matching(topics,spanish_topics)

    cla = utils.get_cross_lingual_alignment(topics,spanish_topics,prepped_text.values,spanish_prepped_text.values)

    other_stats = utils.get_dataset_stats(prepped_text)


    final_object = {"coherence":coherence,"top_tokens":top_tokens,"spanish_coherence":spanish_coherence,"spanish_top_tokens":spanish_top_tokens,
                    "diversity":diversity,"spanish_diversity":spanish_diversity,"average_matching":average_matching,
                    "cla":cla,**CONSTANTS,**other_stats,"vocab_size":tp.vocab_size,"embedding_model":embedding_model,"training_time":training_time}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results_GMM_scientific.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




