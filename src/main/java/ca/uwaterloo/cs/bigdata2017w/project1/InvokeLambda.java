package ca.uwaterloo.cs.bigdata2017w.project1;

import java.nio.ByteBuffer;
import java.nio.charset.Charset;

import com.amazonaws.services.lambda.model.InvocationType;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.regions.Region;
import com.amazonaws.regions.Regions;
import com.amazonaws.services.lambda.AWSLambdaClient;
import com.amazonaws.services.lambda.model.InvokeRequest;

public class InvokeLambda {
    private static final Log logger = LogFactory.getLog(InvokeLambda.class);

    private static final String awsAccessKeyId = "AKIAIW3Q3SNEHZ4TJTQQ";
    private static final String awsSecretAccessKey = "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";
    private static final String regionName = "us-west-2";
    private static final String functionName = "arn:aws:lambda:us-west-2:868710585169:function:yaoleTestFunction";
    private static final String invocationType = "RequestReponse";

    private static Region region;
    private static AWSCredentials credentials;
    private static AWSLambdaClient lambdaClient;

    /**
     * The entry point into the AWS lambda function.
     */
    public static void main(String[] args) {
        credentials = new BasicAWSCredentials(awsAccessKeyId, awsSecretAccessKey);

        lambdaClient = new AWSLambdaClient(credentials);
        region = Region.getRegion(Regions.fromName(regionName));
        lambdaClient.setRegion(region);

        try {
            InvokeRequest invokeRequest = new InvokeRequest();
            invokeRequest.setFunctionName(functionName);
            invokeRequest.setPayload("Test");
            invokeRequest.setInvocationType(InvocationType.fromValue(invocationType));

            System.out.println(byteBufferToString(lambdaClient.invoke(invokeRequest).getPayload(), Charset.forName("UTF-8")));
        } catch (Exception e) {
            logger.error(e.getMessage());
        }
    }

    public static String byteBufferToString(ByteBuffer buffer, Charset charset) {
        byte[] bytes;
        if (buffer.hasArray()) {
            bytes = buffer.array();
        } else {
            bytes = new byte[buffer.remaining()];
            buffer.get(bytes);
        }
        return new String(bytes, charset);
    }
}
