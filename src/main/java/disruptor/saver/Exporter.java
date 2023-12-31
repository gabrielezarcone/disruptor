package disruptor.saver;

import disruptor.costants.FilePaths;
import lombok.Getter;
import lombok.Setter;
import weka.core.Instances;
import weka.core.converters.AbstractFileSaver;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Exporter {

    /**
     * @param saver the saver used by this export
     * @return the saver used by this export
     */
    @Setter @Getter
    private AbstractFileSaver saver;

    /**
     * @param exportedFile the exported file
     * @return the exported file
     */
    @Setter @Getter
    private File exportedFile;

    public Exporter(AbstractFileSaver saver){
        this.saver = saver;
    }

    /**
     * Export the {@link Instances} object specified in instances using the weka saver
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public void export(Instances instances, String outputFilename) throws IOException {
        export(instances, FilePaths.OUTPUT_FOLDER, outputFilename);
    }

    /**
     * Export the {@link Instances} object specified in instances using the weka saver
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param outputDirectoryPath path of the output directory. this
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public void export(Instances instances, String outputDirectoryPath, String outputFilename) throws IOException {

        String fileExtension = null;
        try {
            fileExtension = this.saver.getFileExtension();
        } catch (Exception e) {
            fileExtension = "";
        }

        this.saver.setInstances(instances);
        this.saver.setFile(new File( outputDirectoryPath + outputFilename + fileExtension) );
        this.saver.writeBatch();

        this.exportedFile = saver.retrieveFile();
    }



    /**
     * Export the {@link Instances} object specified in instances using the weka saver appending the today date in the specified format
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param dateFormatPattern format of the date to append after the outputFilename
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public void exportWithDate(Instances instances, String outputFilename, String dateFormatPattern) throws IOException {
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(dateFormatPattern);
        String date = simpleDateFormat.format(new Date());

        outputFilename = outputFilename + date;

        export(instances, outputFilename);
    }



    /**
     * Export the {@link Instances} object specified in instances using the weka saver appending the today date
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public void exportWithDate(Instances instances, String outputFilename) throws IOException {
        String pattern = "yyyy-MM-dd HHmmss";
        exportWithDate(instances, outputFilename, pattern);
    }

    /**
     * Export the {@link Instances} object specified in instances using the weka saver in the specified directory
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param dirName directory inside {@link FilePaths}.OUTPUT_FOLDER
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public void exportInFolder(Instances instances, String dirName, String outputFilename) throws IOException {
        String outputDirectoryPath = dirName + File.separator;
        export(instances, outputDirectoryPath, outputFilename);
    }

    /**
     * Export the {@link Instances} object specified in instances using the weka saver appending the today date in the specified directory
     * <p/>
     * The output destination is specified by the constant {@link FilePaths}.OUTPUT_FOLDER
     * <p/>
     * @param instances {@link Instances} object to export
     * @param dirName directory inside {@link FilePaths}.OUTPUT_FOLDER
     * @param outputFilename filename of the exported file
     * @throws IOException if problem during the export
     */
    public void exportWithDateInFolder(Instances instances, String dirName, String outputFilename) throws IOException {
        String path = dirName + File.separator + outputFilename;
        exportWithDate(instances, path);
    }
}
