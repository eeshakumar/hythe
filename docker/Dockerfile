FROM openjdk:8
FROM python:3.7

ENV PATH /opt/conda/bin:$PATH
ENV DISPLAY :0


# Run installers.
COPY installation /tmp/installation
RUN bash /tmp/installation/install_bazel.sh
RUN bash /tmp/installation/install_python_venv.sh

RUN pip3 install --upgrade pip
RUN pip3 install virtualenv
RUN pip3 install -r /tmp/installation/requirements.txt

# renewe bash
CMD [ "/bin/bash" ]
WORKDIR /hythe
