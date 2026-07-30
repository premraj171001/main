FROM python:3.10-slim
WORKDIR /Healthbridge
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 3012
CMD ['python','app.py']
