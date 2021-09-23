import argparse
from pathlib import Path
from tilse.data.timelines import Timeline as TilseTimeline
from tilse.data.timelines import GroundTruth as TilseGroundTruth
from tilse.evaluation import rouge
from news_tls import utils, data, datewise, clust
from pprint import pprint
from news_tls import summarizers
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer 
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity
def get_scores(metric_desc, pred_tl, groundtruth, evaluator):

    if metric_desc == "concat":
        return evaluator.evaluate_concat(pred_tl, groundtruth)
    elif metric_desc == "agreement":
        return evaluator.evaluate_agreement(pred_tl, groundtruth)
    elif metric_desc == "align_date_costs":
        return evaluator.evaluate_align_date_costs(pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs":
        return evaluator.evaluate_align_date_content_costs(
            pred_tl, groundtruth)
    elif metric_desc == "align_date_content_costs_many_to_one":
        return evaluator.evaluate_align_date_content_costs_many_to_one(
            pred_tl, groundtruth)


def zero_scores():
    return {'f_score': 0., 'precision': 0., 'recall': 0.}


def evaluate_dates(pred, ground_truth):
    pred_dates = pred.get_dates()
    ref_dates = ground_truth.get_dates()
    shared = pred_dates.intersection(ref_dates)
    n_shared = len(shared)
    n_pred = len(pred_dates)
    n_ref = len(ref_dates)
    prec = n_shared / n_pred
    rec = n_shared / n_ref
    if prec + rec == 0:
        f_score = 0
    else:
        f_score = 2 * prec * rec / (prec + rec)
    return {
        'precision': prec,
        'recall': rec,
        'f_score': f_score,
    }


def get_average_results(tmp_results):
    rouge_1 = zero_scores()
    rouge_2 = zero_scores()
    date_prf = zero_scores()
    for rouge_res, date_res, _ in tmp_results:
        metrics = [m for m in date_res.keys() if m != 'f_score']
        for m in metrics:
            rouge_1[m] += rouge_res['rouge_1'][m]
            rouge_2[m] += rouge_res['rouge_2'][m]
            date_prf[m] += date_res[m]
    n = len(tmp_results)
    for result in [rouge_1, rouge_2, date_prf]:
        for k in ['precision', 'recall']:
            result[k] /= n
        prec = result['precision']
        rec = result['recall']
        if prec + rec == 0:
            result['f_score'] = 0.
        else:
            result['f_score'] = (2 * prec * rec) / (prec + rec)
    return rouge_1, rouge_2, date_prf


def evaluate(tls_model, dataset, result_path, trunc_timelines=False, time_span_extension=0):

    results = []
    n_topics = len(dataset.collections)
    vec=TfidfVectorizer(stop_words='english',decode_error="ignore",token_pattern = r'\b\w+\b',min_df=1)
    for i, collection in enumerate(dataset.collections):

        ref_timelines = [TilseTimeline(tl.date_to_summaries)
                         for tl in collection.timelines]
        topic = collection.name
        n_ref = len(ref_timelines)

        if trunc_timelines:
            ref_timelines = data.truncate_timelines(ref_timelines, collection)

        for j, ref_timeline in enumerate(ref_timelines):
            print(f'topic {i+1}/{n_topics}: {topic}, ref timeline {j+1}/{n_ref}')
            if ((topic=="bpoil"and j==2)or(topic=="finan"and j==0)):
                raw_sents=[]
                for a in collection.articles():
                    for s in a.sentences:
                        raw_sents.append(s.raw)
                ngram_vectorizer = TfidfVectorizer(stop_words='english',decode_error="ignore",token_pattern = r'\b\w+\b',min_df=1)
                try:
                    x1 = ngram_vectorizer.fit_transform(raw_sents)
                except:
                    return None
                ngram_freq=x1.toarray().sum(axis=0)
                ngram_vectorizer1 = CountVectorizer(stop_words='english', decode_error="ignore",token_pattern = r'\b\w+\b',min_df=1)
                tls_model.load(ignored_topics=[collection.name])

                ref_dates = sorted(ref_timeline.dates_to_summaries)

                start, end = data.get_input_time_span(ref_dates, time_span_extension)

                collection.start = start
                collection.end = end

                #utils.plot_date_stats(collection, ref_dates)

                l = len(ref_dates)
                k = data.get_average_summary_length(ref_timeline)

                pred_timeline_ = tls_model.predict(
                    collection,
                    max_dates=l,
                    max_summary_sents=k,
                    ref_tl=ref_timeline # only oracles need this
                )
                print('timeline done')
                pred_timeline = TilseTimeline(pred_timeline_.date_to_summaries)
                date1=[]
                datelist=[]
                datesum=0
                for date in sorted(pred_timeline.dates_to_summaries.keys()):
                    date1.append(str(date))
                    tx2=[]
                    tsum=0
                    for sent in pred_timeline.dates_to_summaries[date]:
                        tx2.append(sent)
                    x2=ngram_vectorizer1.fit_transform(tx2)
                    ngram1_freq=x2.toarray().sum(axis=0)
                    for each in ngram_vectorizer1.vocabulary_:
                        try:
                            index=ngram_vectorizer.get_feature_names().index(each)
                            index1=ngram_vectorizer1.vocabulary_.get(each)
                            tsum+=(ngram_freq[index]*ngram1_freq[index1] if index and index1 else 0 )
                        except ValueError:
                            pass
                    datesum+=tsum
                    datelist.append(tsum)
                datelist=datelist/datesum
                f = open(f'out-guding-our-imp-t-{topic}-{j}.txt', "w")
                print(dict(zip(date1,datelist)),file=f)
def main(args):

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {args.dataset}')
    dataset = data.Dataset(dataset_path)
    dataset_name = dataset_path.name

    if args.method == 'datewise':
        resources = Path(args.resources)
        models_path = resources / 'supervised_date_ranker.{}.pkl'.format(
            dataset_name
        )
        # load regression models for date ranking
        key_to_model = utils.load_pkl(models_path)
        date_ranker = datewise.SupervisedDateRanker(method='regression')
        sent_collector = datewise.PM_Mean_SentenceCollector(
            clip_sents=5, pub_end=2)
        summarizer = summarizers.SubmodularSummarizer()
        system = datewise.DatewiseTimelineGenerator(
            date_ranker=date_ranker,
            summarizer=summarizer,
            sent_collector=sent_collector,
            key_to_model = key_to_model
        )

    elif args.method == 'clust':
        cluster_ranker = clust.ClusterDateMentionCountRanker()
        clusterer = clust.TemporalMarkovClusterer()
        summarizer = summarizers.CentroidOpt()
        system = clust.ClusteringTimelineGenerator(
            cluster_ranker=cluster_ranker,
            clusterer=clusterer,
            summarizer=summarizer,
            clip_sents=5,
            unique_dates=True,
        )
    else:
        raise ValueError(f'Method not found: {args.method}')


    if dataset_name == 'entities':
        evaluate(system, dataset, args.output, trunc_timelines=True, time_span_extension=7)
    else:
        evaluate(system, dataset, args.output, trunc_timelines=False, time_span_extension=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--resources', default=None,
        help='model resources for tested method')
    parser.add_argument('--output', default=None)
    main(parser.parse_args())
