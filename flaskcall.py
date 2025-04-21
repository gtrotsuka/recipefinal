import requests
import csv

res = requests.post('http://127.0.0.1:5000/rank', json={
    'user_ingredients': ['beef', 'onion', 'egg', 'flour', 'butter', 'potatoes']
})
print(res.json())
with open('test.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(res.json().keys())
    writer.writerow(res.json().values())
