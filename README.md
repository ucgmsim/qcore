[![Build Status](https://travis-ci.org/ucgmsim/qcore.svg?branch=master)](https://travis-ci.org/ucgmsim/qcore)

Migration to Jenkins (from Travis-CI)
1. Removed .travis.yml
2. Removed Webhooks from the repository setting
3. Removed the repository link from Travis-CI web
4. Added "jenkins" user and group to Docker image, so that the Docker image can be run not as root - preventing lots of files with root permission that can't be cleaned without sudo.
5. Jenkinsfile now controls the workflow of auto-testing


