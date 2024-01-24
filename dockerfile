#
# Build stage
#
FROM maven:3.9.6-eclipse-temurin-8 AS build
ENV HOME=/usr/app
RUN mkdir -p $HOME
WORKDIR $HOME
ADD . $HOME
RUN --mount=type=cache,target=/root/.m2 mvn -f $HOME/pom.xml clean package

#
# Package stage
#
FROM eclipse-temurin:8-jre-jammy
ARG JAR_FILE=/usr/app/target/deploy/disruptor/*.jar
COPY --from=build $JAR_FILE /app/runner.jar
ENTRYPOINT java -jar /app/runner.jar