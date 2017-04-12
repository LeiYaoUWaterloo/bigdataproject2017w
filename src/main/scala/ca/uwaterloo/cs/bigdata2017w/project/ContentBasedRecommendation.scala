package ca.uwaterloo.cs.bigdata2017w.project

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._


import scala.collection.mutable._
import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import io.bespin.scala.util.Tokenizer

import scala.collection.mutable

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

  def wcIter(iter: Iterator[String]): Iterator[(String, Int)] = {
    val counts = new HashMap[String, Int]() { override def default(key: String) = 0 }

    iter.flatMap(line => tokenize(line))
      .foreach { t => counts.put(t, counts(t) + 1) }

    counts.iterator
  }

  val mapper = new ObjectMapper()
  mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
  mapper.registerModule(DefaultScalaModule)

  def main(argv: Array[String]): Unit = {
    val args = new Conf(argv)

    log.info("Input: " + args.review())
    log.info("Output: " + args.output())

    val conf = new SparkConf().setAppName("ContentBasedRecommendation")
    val sc = new SparkContext(conf)

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    //construct stop words hashset
    val stopWords = sc.broadcast(scala.io.Source.fromFile("data/stopwords.txt").getLines().toSet).value

    //construct stemming words hashmap
    val stemming_word = sc.textFile("data/result.txt")
    val map2 = stemming_word.map(line =>{
      val words = tokenize(line)
      (words(0), words(1))
    }).collectAsMap()
    val stemmer = sc.broadcast(map2)

    val review = sc.textFile(args.review())
    var results = review.flatMap(record => {
      Some(mapper.readValue(record, classOf[Review]))
    }).map(array => {
      (array.business_id, array.text)
    }).groupByKey()
      .map(tuple => {
        val reviews = new ArrayBuffer[String]()
        val iter = tuple._2.iterator
        while(iter.hasNext) {
          val words = tokenize(iter.next())
          for (word <- words) {
            val wordLowcase = word.toLowerCase
            if (!stopWords.contains(wordLowcase)) {
              if (stemmer.value.contains(wordLowcase)) {
                reviews += stemmer.value(wordLowcase)
              } else {
                reviews += wordLowcase
              }
            }
          }
        }
        (tuple._1, reviews)
      }).map(tuple => {
      val termFreqs = tuple._2.foldLeft(new mutable.HashMap[String, Int]()) {
        (map, term) => {
          map += term -> (map.getOrElse(term, 0) + 1)
          map
        }
      }
      termFreqs
    })


/*      .flatMap(line => {
      tokenize(line)
    }).filter(word => {
      val wordLowcase = word.toLowerCase
      !stopWords.contains(wordLowcase)
    }).map(word => {
      val wordLowcase = word.toLowerCase
      if (stemmer.value.contains(wordLowcase)) {
        stemmer.value(wordLowcase)
      } else {
        wordLowcase
      }
    }).mapPartitions(wcIter)
      .reduceByKey(_ + _)
      .map(tuple => tuple.swap)
      .sortByKey(false)
      .map(tuple => tuple.swap)
      .filter(tuple => {
        tuple._2 > 2
      })
*/
  }
}