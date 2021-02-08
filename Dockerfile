FROM python:3.7
ADD cli.py /

RUN pip install PyInquirer
RUN pip install pandas
RUN pip install numpy
RUN pip install scikit-learn
RUN pip install matplotlib
RUN pip install	onnxruntime
RUN mkdir -p /usr/src/app

CMD [ "python", "./cli.py" ]
