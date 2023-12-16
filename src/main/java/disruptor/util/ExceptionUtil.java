package disruptor.util;

import org.slf4j.Logger;

public class ExceptionUtil {
    private ExceptionUtil(){}

    /**
     * Uniformed logging for all the exceptions
     * @param e exception to log
     * @param log logger used for logging
     */
    public static void logException(Exception e, Logger log){
        log.error("\tException: {}", e.getClass().getSimpleName());
        if(e.getMessage() != null){
            log.error("\tMessage: {}", e.getMessage());
        }
        if(e.getCause() != null){
            log.error("\tCause: {}", e.getCause().toString());
        }
        if(log.isTraceEnabled()){
            log.trace("Stack Trace: ");
            e.printStackTrace();
        }
        log.error("---------------------------------------------------------------");
    }
}
