from python-3.10
WORKDIR /Healthbridge
COPY ./requirements.txt
RUN pip install requirements.txt
copy . .
expose 3012
CMD ['python','app.py']
