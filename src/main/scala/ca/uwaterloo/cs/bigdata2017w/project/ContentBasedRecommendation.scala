package ca.uwaterloo.cs.bigdata2017w.project

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Matrix

import scala.collection.mutable._
import scala.collection.mutable
import scala.collection.JavaConversions._
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import io.bespin.scala.util.Tokenizer

class Conf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(review, output)
  val review = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  verify()
}

class Review {
  @JsonProperty var review_id: String = null
  @JsonProperty var user_id: String = null
  @JsonProperty var business_id: String = null
  @JsonProperty var stars: String = null
  @JsonProperty var date: String = null
  @JsonProperty var text: String = null
  @JsonProperty var useful: String = null
  @JsonProperty var funny: String = null
  @JsonProperty var cool: String = null

  override def toString = s"Review(user_id=$user_id, business_id=$business_id, stars=$stars, " +
    s"date=$date, text=$text, useful=$useful, funny=$funny, cool=$cool)"
}

object ContentBasedRecommendation extends Tokenizer{
  val log = Logger.getLogger(getClass().getName())

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  val mapper = new ObjectMapper()
  mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  mapper.registerModule(DefaultScalaModule)

  def main(argv: Array[String]): Unit = {
    //set up
    val args = new Conf(argv)

    log.info("Input: " + args.review())

    val conf = new SparkConf().setAppName("ContentBasedRecommendation")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    //construct stop words hashmap
    val stop_word = sc.textFile("data/stopwords.txt")
    val map1 = stop_word.map(line =>{
      (line, 1)
    }).collectAsMap()
    val stopWords = sc.broadcast(map1).value

    //construct stemming words hashmap
    val stemming_word = sc.textFile("data/result.txt")
    val map2 = stemming_word.map(line =>{
      val words = tokenize(line)
      (words(0), words(1))
    }).collectAsMap()
    val stemmer = sc.broadcast(map2).value

    val review = sc.textFile(args.review())

    //aggregate reviews for each bussiness
    val aggregatedReview = review.flatMap(record => {
      Some(mapper.readValue(record, classOf[Review]))
    }).map(array => {
      (array.business_id, array.text)
    }).groupByKey()

    //preprocess: tokenize + lowcase + delele stopwords + only words + stemming
    val preprocessedReviews = aggregatedReview.map(businessIdReviews => {
        val reviews = new ArrayBuffer[String]()
        val iter = businessIdReviews._2.iterator
        while(iter.hasNext) {
          val words = tokenize(iter.next())
          for (word <- words) {
            val wordLowcase = word.toLowerCase
            if (!stopWords.containsKey(wordLowcase) && isOnlyLetters(wordLowcase)) {
              if (stemmer.containsKey(wordLowcase)) {
                reviews += stemmer(wordLowcase)
              } else {
                reviews += wordLowcase
              }
            }
          }
        }
        (businessIdReviews._1, reviews)
      })

    //business and a hashmap (term of this business -> term frequency)
    val businessTermFreqs = preprocessedReviews.map(businessIdReviews => {
      val termFreqs = businessIdReviews._2.foldLeft(new mutable.HashMap[String, Int]()) {
        (map, term) => {
          map += term -> (map.getOrElse(term, 0) + 1)
          map
        }
      }
      (businessIdReviews._1, termFreqs)
    })
    businessTermFreqs.cache()

    //total number of bussinesses
    val numBusinesses = businessTermFreqs.count()

    //business frequency of each term
    val termDocFreqs = businessTermFreqs.flatMap(businessIdTermsMap => {
      businessIdTermsMap._2.keySet
    }).map((_, 1))
      .reduceByKey(_+_)

    //only take top "numTerms" frequent terms
    val numTerms = 50000
    val ordering = Ordering.by[(String, Int), Int](_._2)
    val topTermDocFreqs = termDocFreqs.top(numTerms)(ordering)

    //compute inverse document frequency for each term
    val idfs = topTermDocFreqs.map{
      case (term, count) => (term, math.log(numBusinesses.toDouble / count))
    }.toMap
    val bIdfs = sc.broadcast(idfs).value

    // [term, index] index is a number indicating which column this term represents
    val termIds = idfs.keys.zipWithIndex.toMap
    val bTermIds = sc.broadcast(termIds).value

    //compute tf-idf and get the final business * term matrix, which is "numBusiness * numTerms"
    val businessIdVecs = businessTermFreqs.map(businessIdTermFreqs => {
      val businessTotalTerms = businessIdTermFreqs._2.values.sum
      val termScores = businessIdTermFreqs._2.filter {
        case (term, freq) => bTermIds.containsKey(term)
      }.map{
        case (term, freq) => (bTermIds(term), businessIdTermFreqs._2(term) * bIdfs(term) / businessTotalTerms)
      }.toSeq
      (businessIdTermFreqs._1, Vectors.sparse(bTermIds.size, termScores))
    })
      businessIdVecs.take(30).foreach(println(_))
      sc.parallelize(businessIdVecs.take(1000)).saveAsTextFile(args.output())

  /*
    val vecs = businessIdVecs.map(businessIdTermTfidf => {
      businessIdTermTfidf._2
    })
    vecs.cache()

    val mat: RowMatrix = new RowMatrix(vecs)
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(500, computeU = true)
    val U: RowMatrix = svd.U
    val s: Vector = svd.s
    val V: Matrix = svd.V
*/
  }
}