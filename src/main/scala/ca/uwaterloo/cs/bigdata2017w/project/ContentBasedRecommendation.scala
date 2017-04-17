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
import org.apache.spark.mllib.linalg.{Matrices, Vector => MLLibVector}

import scala.collection.mutable._
import scala.collection.mutable
import scala.collection.JavaConversions._
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import io.bespin.scala.util.Tokenizer

class Conf(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(user_id, state, review, business, output)

  val user_id = opt[String](descr = "user id", required = true)
  val state = opt[String](descr = "user state", required = true)

  val review = opt[String](descr = "review data", required = true)
  val business = opt[String](descr = "business data", required = true)

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

class Business{
  @JsonProperty var business_id: String = null
  @JsonProperty var stars: String = null
  @JsonProperty var city: String = null
  @JsonProperty var state: String = null
  @JsonProperty var review_count: String = null
  override def toString = s"Business(business_id=$business_id, stars=$stars,city=$city，state=$state, review_count=$review_count)"

}

class User{
  @JsonProperty var user_id: String = null
  @JsonProperty var name: String = null
  @JsonProperty var review_count: String = null
  override def toString = s"User(user_id=$user_id, name=$name, review_count=$review_count)"
}

object ContentBasedRecommendation extends Tokenizer{
  val log = Logger.getLogger(getClass().getName())

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }


  //Finds the product of a distributed matrix and a diagonal matrix represented by a vector
  def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map { vec =>
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    })
  }

  //Returns a matrix where each row is divided by its length.
  def distributedRowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map { vec =>
      val array = vec.toArray
      val length = math.sqrt(array.map(x => x * x).sum)
      Vectors.dense(array.map(_ / length))
    })
  }

  val mapper = new ObjectMapper()
  mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  mapper.registerModule(DefaultScalaModule)

  def main(argv: Array[String]): Unit = {
    val args = new Conf(argv)

    log.info("User id: " + args.user_id())
    log.info("User state: " + args.state())
    log.info("Review data: " + args.review())
    log.info("Business data: " + args.business())

    val conf = new SparkConf().setAppName("ContentBasedRecommendation")
    val sc = new SparkContext(conf)

    //if output directory exits, delete it
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

    //read in all data and transform into RDDs
    val review = sc.textFile(args.review())
    val business = sc.textFile(args.business())
    val user_id =args.user_id()
    val state = args.state()

    //transform json RDD to usual RDD (all businesses information)
    val businessInfo = business.flatMap(record => {
      Some(mapper.readValue(record, classOf[Business]))
    })

    //construct business hashmap given input state
    val map3 = businessInfo.map(array => {
      (array.business_id, array.state)
    }).filter(array =>{
      array._2 == state
    }).collectAsMap()
    val filteredBusiness = sc.broadcast(map3).value

    //aggregate reviews for each bussiness in a state
    val aggregatedReview = review.flatMap(record => {
      Some(mapper.readValue(record, classOf[Review]))
    }).filter(array => {
      filteredBusiness.containsKey(array.business_id)
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
            val wordLowercase = word.toLowerCase
            if (!stopWords.containsKey(wordLowercase) && isOnlyLetters(wordLowercase)) {
              if (stemmer.containsKey(wordLowercase)) {
                reviews += stemmer(wordLowercase)
              } else {
                reviews += wordLowercase
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

    val docIds = businessTermFreqs.map(businessIdTermFreqs => {
      businessIdTermFreqs._1
    }).zipWithUniqueId().collectAsMap()
    val bDocIds = sc.broadcast(docIds).value
    val bIdDoc = sc.broadcast(docIds.map(_.swap)).value

    //total number of bussinesses
    val numBusinesses = businessTermFreqs.count()

    //business frequency of each term
    val termDocFreqs = businessTermFreqs.flatMap(businessIdTermsFreqsMap => {
      businessIdTermsFreqsMap._2.keySet
    }).map((_, 1))
      .reduceByKey(_+_)

    //only take top "numTerms" frequent terms
    val numTerms = 10000
    val ordering = Ordering.by[(String, Int), Int](_._2)
    val topTermDocFreqs = termDocFreqs.top(numTerms)(ordering)

    //compute inverse document frequency (idf) for each term
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

    val vecs = businessIdVecs.map(businessIdTermTfidf => {
      businessIdTermTfidf._2
    })
    vecs.cache()

    val mat: RowMatrix = new RowMatrix(vecs)
    val k = 200
    val svd: SingularValueDecomposition[RowMatrix, Matrix] = mat.computeSVD(k, computeU = true)
    val U: RowMatrix = svd.U
    val s: Vector = svd.s
    val V: Matrix = svd.V

    val collect = U.rows.collect()
    println("U factor is:")
    collect.foreach { vector => println(vector) }
    println(s"Singular values are: $s")
    println(s"V factor is:\n$V")

    val docId = bDocIds("tggHJ7wk-6Wok_CSPd3aUA")
    val US: RowMatrix = multiplyByDiagonalRowMatrix(U, s)
    val normalizedUS: RowMatrix = distributedRowsNormalized(US)

    // Look up the row in US corresponding to the business ID
    val docRowArr = normalizedUS.rows.zipWithUniqueId.map(_.swap).lookup(docId).head.toArray
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every business
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the businesses with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    // Businesses can end up with NaN score if their row in U is all zeros.  Filter these out.
    val idWeights = allDocWeights.filter(!_._1.isNaN).top(10)

    println(idWeights.map { case (score, id) => (bIdDoc(id), score) }.mkString(", "))

  }
}