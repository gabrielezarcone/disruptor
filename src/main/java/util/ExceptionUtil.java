package util;

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
        log.error("\tCause: {}", e.getCause().toString());
        log.trace("Stack Trace: ", e);
        log.error("---------------------------------------------------------------");
    }
}
