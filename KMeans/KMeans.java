//17341073 �����
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import java.util.Arrays;
import java.util.Iterator;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.util.LineReader;

public class KMeans{
    public static class Map extends Mapper<LongWritable, Text, IntWritable, Text>{
        ArrayList<ArrayList<Double>> centers = null;
        //k������
        int k = 0;
        //��ȡ����
        protected void setup(Context context) throws IOException,
                InterruptedException {
            centers = KMeans.getCenter(context.getConfiguration().get("centersPath"),false);
            k = centers.size();
        }
        /*
         * 1.ÿ�ζ�ȡһ��Ҫ���������¼���������Աȣ����ൽ��Ӧ������
         * 2.������IDΪkey�����İ����ļ�¼Ϊvalue���
         */
        protected void map(LongWritable key, Text value, Context context)throws IOException, InterruptedException {
            //��ȡһ������
            ArrayList<Double> fileds = KMeans.textToArray(value);
            int sizeOfFileds = fileds.size();
            double minDistance = 99999999;
            int centerIndex = 0;
            //����ȡ��k�����ĵ��뵱ǰ��ȡ�ļ�¼������,ѭ���ҳ�����ü�¼��ӽ������ĵ��ID
            for(int i=0;i<k;i++){
                double currentDistance = 0;
                for(int j=0;j<sizeOfFileds;j++){
                    double centerPoint = Math.abs(centers.get(i).get(j));
                    double filed = Math.abs(fileds.get(j));
                    currentDistance += Math.pow((centerPoint - filed) / (centerPoint + filed), 2);
                }
                if(currentDistance<minDistance){
                    minDistance = currentDistance;
                    centerIndex = i;
                }
            }
            context.write(new IntWritable(centerIndex+1), value);
        }
    }
    
    //����reduce�Ĺ鲢����������ΪKey����¼�鲢��һ��
    public static class Reduce extends Reducer<IntWritable, Text, Text, Text>{
        /*
         * 1.KeyΪ�������ĵ�ID valueΪ�����ĵļ�¼����
         * 2.�������м�¼Ԫ�ص�ƽ��ֵ������µ�����
         */
        protected void reduce(IntWritable key, Iterable<Text> value,Context context)
                throws IOException, InterruptedException {
            ArrayList<ArrayList<Double>> filedsList = new ArrayList<ArrayList<Double>>();
            //���ζ�ȡ��¼����ÿ��Ϊһ��ArrayList<Double>
            for(Iterator<Text> it =value.iterator();it.hasNext();){
                ArrayList<Double> tempList = KMeans.textToArray(it.next());
                filedsList.add(tempList);
            }
            //�����µ�����
            int filedSize = filedsList.get(0).size();
            double[] avg = new double[filedSize];
            for(int i=0;i<filedSize;i++){
                //��ÿ�е�ƽ��ֵ
                double sum = 0;
                int size = filedsList.size();
                for(int j=0;j<size;j++){
                    sum += filedsList.get(j).get(i);
                }
                avg[i] = sum / size;
            }
            context.write(new Text("") , new Text(Arrays.toString(avg).replace("[", "").replace("]", "")));
        }
    }
    @SuppressWarnings("deprecation")
    public static void start(String centerPath,String dataPath,String newCenterPath,boolean runReduce) throws IOException, ClassNotFoundException, InterruptedException{
        Configuration conf = new Configuration();
        conf.set("centersPath", centerPath);
        Job job = new Job(conf, "mykmeans");
        job.setJarByClass(KMeans.class);
        job.setMapperClass(Map.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        if(runReduce){
            job.setReducerClass(Reduce.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);
        }
        FileInputFormat.addInputPath(job, new Path(dataPath));
        FileOutputFormat.setOutputPath(job, new Path(newCenterPath));
        System.out.println(job.waitForCompletion(true));
    }

    public static void main(String[] args) throws ClassNotFoundException, IOException, InterruptedException {
        String centerPath = "hdfs://master:9000/input/oldcenter.txt";
        String dataPath = "hdfs://master:9000/input/data.txt";
        String newCenterPath = "hdfs://master:9000/output1";
        int count = 0;
        while(true){
            start(centerPath,dataPath,newCenterPath,true);
            System.out.println(" �� " + ++count + " �μ��� ");
            if(KMeans.compare(centerPath,newCenterPath )){
                start(centerPath,dataPath,newCenterPath,false);
                break;
            }
        }
    }
        
        //��ȡ�����ļ�������
        public static ArrayList<ArrayList<Double>> getCenter(String centersPath,boolean isDirectory) throws IOException{
            ArrayList<ArrayList<Double>> result = new ArrayList<ArrayList<Double>>();
            Path path = new Path(centersPath);
            Configuration conf = new Configuration();
            FileSystem fileSystem = path.getFileSystem(conf);
            if(isDirectory){    
                FileStatus[] listFile = fileSystem.listStatus(path);
                for (int i = 0; i < listFile.length; i++) {
                    result.addAll(getCenter(listFile[i].getPath().toString(),false));
                }
                return result;
            }
            FSDataInputStream fsis = fileSystem.open(path);
            LineReader lineReader = new LineReader(fsis, conf);
            Text line = new Text();
            while(lineReader.readLine(line) > 0){
                ArrayList<Double> tempList = textToArray(line);
                result.add(tempList);
            }
            lineReader.close();
            return result;
        }
        //ɾ���ļ�
        public static void deletePath(String pathStr) throws IOException{
            Configuration conf = new Configuration();
            Path path = new Path(pathStr);
            FileSystem hdfs = path.getFileSystem(conf);
            hdfs.delete(path ,true);
        }
        public static ArrayList<Double> textToArray(Text text){
            ArrayList<Double> list = new ArrayList<Double>();
            String[] fileds = text.toString().split(",");
            for(int i=0;i<fileds.length;i++){
                list.add(Double.parseDouble(fileds[i]));
            }
            return list;
        }
        public static boolean compare(String centerPath,String newPath) throws IOException{
            List<ArrayList<Double>> oldCenters = KMeans.getCenter(centerPath,false);
            List<ArrayList<Double>> newCenters = KMeans.getCenter(newPath,true);
            int size = oldCenters.size();
            int fildSize = oldCenters.get(0).size();
            double distance = 0;
            for(int i=0;i<size;i++){
                for(int j=0;j<fildSize;j++){
                    double t1 = Math.abs(oldCenters.get(i).get(j));
                    double t2 = Math.abs(newCenters.get(i).get(j));
                    distance += Math.pow((t1 - t2) / (t1 + t2), 2);
                }
            }
            if(distance == 0.0){
                //ɾ���µ������ļ��Ա�������ι������
                KMeans.deletePath(newPath);
                return true;
            }else{
                //����������ļ������µ������ļ����Ƶ������ļ��У���ɾ�������ļ�
                Configuration conf = new Configuration();
                Path outPath = new Path(centerPath);
                FileSystem fileSystem = outPath.getFileSystem(conf);
                FSDataOutputStream overWrite = fileSystem.create(outPath,true);
                overWrite.writeChars("");
                overWrite.close();
                Path inPath = new Path(newPath);
                FileStatus[] listFiles = fileSystem.listStatus(inPath);
                for (int i = 0; i < listFiles.length; i++) {                
                    FSDataOutputStream out = fileSystem.create(outPath);
                    FSDataInputStream in = fileSystem.open(listFiles[i].getPath());
                    IOUtils.copyBytes(in, out, 4096, true);
                }
                //ɾ���µ������ļ��Ա�ڶ��������������
                KMeans.deletePath(newPath);
            }
            return false;
        }
}