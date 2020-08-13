import requests
from bs4 import BeautifulSoup as bs
from datetime import datetime
import pandas as pd

api = 'https://aspm.faa.gov/opsnet/sys/opsnet-server-x.asp'

# POST headers
headers = {
	"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36",
	'Cookie': 'ASPMFAAGOV=OPSNET%5FLEVEL=%2D1; ASPSESSIONIDCATTCATT=PFHBEMHDDKCMKMMDGBKCEKCN'
}
payload = {
	'dstyle': 'm',
	'dfld': 'yyyymm',
	'fromdate': '202001',
	'todate': '202001', 
	'keylist': 'yyyymm,LOCID,REGION,DDSO_SA,CLASS_ID,STATE',
	'line': 'SELECT YYYYMM,LOCID,REGION,DDSO_SA,CLASS_ID,STATE ,SUM(IFR_ITIN_AC) AS IFR_ITIN_AC,SUM(IFR_ITIN_AT) AS IFR_ITIN_AT,SUM(IFR_ITIN_GA) AS IFR_ITIN_GA,SUM(IFR_ITIN_MI) AS IFR_ITIN_MI,SUM(IFR_ITIN_AC+IFR_ITIN_AT+IFR_ITIN_GA+IFR_ITIN_MI) AS IFR_ITIN_TOT,SUM(IFR_OVER_AC) AS IFR_OVER_AC,SUM(IFR_OVER_AT) AS IFR_OVER_AT,SUM(IFR_OVER_GA) AS IFR_OVER_GA,SUM(IFR_OVER_MI) AS IFR_OVER_MI,SUM(IFR_OVER_AC+IFR_OVER_AT+IFR_OVER_GA+IFR_OVER_MI) AS IFR_OVER_TOT,SUM(VFR_ITIN_AC) AS VFR_ITIN_AC,SUM(VFR_ITIN_AT) AS VFR_ITIN_AT,SUM(VFR_ITIN_GA) AS VFR_ITIN_GA,SUM(VFR_ITIN_MI) AS VFR_ITIN_MI,SUM(VFR_ITIN_AC+VFR_ITIN_AT+VFR_ITIN_GA+VFR_ITIN_MI) AS VFR_ITIN_TOT,SUM(VFR_OVER_AC) AS VFR_OVER_AC,SUM(VFR_OVER_AT) AS VFR_OVER_AT,SUM(VFR_OVER_GA) AS VFR_OVER_GA,SUM(VFR_OVER_MI) AS VFR_OVER_MI,SUM(VFR_OVER_AC+VFR_OVER_AT+VFR_OVER_GA+VFR_OVER_MI) AS VFR_OVER_TOT,SUM(CIVIL) AS LOCAL_GA,SUM(LOCAL_MI) AS LOCAL_MI,SUM(CIVIL+LOCAL_MI) AS TOT_LOC,SUM(ARPT_OPS) AS ARPT_OPS,SUM(TOW_OPS) AS TOW_OPS  FROM TOWER_DAY WHERE YYYYMM>=202001 AND YYYYMM<=202001 GROUP BY YYYYMM,LOCID,REGION,DDSO_SA,CLASS_ID,STATE ORDER BY YYYYMM,LOCID,REGION,DDSO_SA,CLASS_ID,STATE',
	'cmd': 'tow_bas',
	'nopage': 'y',
	'nost': 'n',
	'avgdays': '1',
	'oktosave': 'y',
	'additiifr': 'y',
	'additivfr': 'y',
	'addoveifr': 'y',
	'addovevfr': 'y',
	'addloc': 'y',
	'facilityType': 'l',
	'locMode': 'on',
	'dtype': 'm',
	'fm_m': '01',
	'fy_m': '2020',
	'tm_m': '01',
	'ty_m': '2020',
	'daytype': 'all',
	'fy_y': '2020',
	'ytype': 'c',
	'ty_y': '2020',
	'ydaytype': 'all',
	'fm_r': '01',
	'fd_r': '01',
	'fy_r': '2020',
	'tm_r': '01',
	'td_r': '01',
	'ty_r': '2020',
	'rdaytype': 'all',
	'compdtype': 'd',
	'compfm_m': '01',
	'compfy_m': '2020',
	'comptm_m': '01',
	'compty_m': '2020',
	'compdaytype': 'all',
	'compfy_y': '2020',
	'compytype': 'c',
	'compty_y': '2020',
	'ycompdaytype': 'all',
	'compfm_r': '01',
	'compfd_r': '01',
	'compfy_r': '2020',
	'comptm_r': '01',
	'comptd_r': '01',
	'compty_r': '2020',
	'rcompdaytype': 'all',
	'complogical': '{YDAY}',
	'ftype': '0',
	'reptype': 'bas',
	'rank': 'tow_ops',
	'miss': 'p',
	'peak': 'TOW_OPS',
	'topcnt': '5',
	'iti_ifr': '1',
	'iti_vfr': '1',
	'ove_ifr': '1',
	'ove_vfr': '1',
	'loc': '1',
	'reportformat': 'asp'
}

# Setup Dataframe
higher_categories = ['IFR Itinerant', 'IFR Overflight', 'VFR Itinerant', 'VFR Overflight']
sub_categories = ['Air Carrier', 'Air Taxi', 'General Aviation', 'Military', 'Total']
columns = ['Date', 'Facility', 'Region', 'DDSO Service Area', 'Class', 'State']
for h in higher_categories:
	for s in sub_categories:
		columns.append(h + ' - ' + s)
columns += ['Local - Civil', 'Local - Military', 'Local - Total']
columns += ['Airport Operations', 'Tower Operations']
contents = [[] for _ in range(len(columns))]

# Setup dates
currentMonth = datetime.now().month
currentYear = datetime.now().year
years = [str(y) for y in range(1989, currentYear + 1)]
months = [str(m) for m in range(1, 13)]

# Send requests and concatenate data
for year in years:
	if year == currentYear:
		months = [str(m) for m in range(1, currentMonth + 1)]
	for month in months:
		if len(month) == 1:
			month = '0' + month
		ym = year + month
		payload['fromdate'] = ym
		payload['todate'] = ym
		payload['line'] = "SELECT YYYYMM,LOCID,REGION,DDSO_SA,CLASS_ID,STATE ,SUM(IFR_ITIN_AC) AS IFR_ITIN_AC,SUM(IFR_ITIN_AT) AS IFR_ITIN_AT,SUM(IFR_ITIN_GA) AS IFR_ITIN_GA,SUM(IFR_ITIN_MI) AS IFR_ITIN_MI,SUM(IFR_ITIN_AC+IFR_ITIN_AT+IFR_ITIN_GA+IFR_ITIN_MI) AS IFR_ITIN_TOT,SUM(IFR_OVER_AC) AS IFR_OVER_AC,SUM(IFR_OVER_AT) AS IFR_OVER_AT,SUM(IFR_OVER_GA) AS IFR_OVER_GA,SUM(IFR_OVER_MI) AS IFR_OVER_MI,SUM(IFR_OVER_AC+IFR_OVER_AT+IFR_OVER_GA+IFR_OVER_MI) AS IFR_OVER_TOT,SUM(VFR_ITIN_AC) AS VFR_ITIN_AC,SUM(VFR_ITIN_AT) AS VFR_ITIN_AT,SUM(VFR_ITIN_GA) AS VFR_ITIN_GA,SUM(VFR_ITIN_MI) AS VFR_ITIN_MI,SUM(VFR_ITIN_AC+VFR_ITIN_AT+VFR_ITIN_GA+VFR_ITIN_MI) AS VFR_ITIN_TOT,SUM(VFR_OVER_AC) AS VFR_OVER_AC,SUM(VFR_OVER_AT) AS VFR_OVER_AT,SUM(VFR_OVER_GA) AS VFR_OVER_GA,SUM(VFR_OVER_MI) AS VFR_OVER_MI,SUM(VFR_OVER_AC+VFR_OVER_AT+VFR_OVER_GA+VFR_OVER_MI) AS VFR_OVER_TOT,SUM(CIVIL) AS LOCAL_GA,SUM(LOCAL_MI) AS LOCAL_MI,SUM(CIVIL+LOCAL_MI) AS TOT_LOC,SUM(ARPT_OPS) AS ARPT_OPS,SUM(TOW_OPS) AS TOW_OPS  FROM TOWER_DAY WHERE YYYYMM>={} AND YYYYMM<={} GROUP BY YYYYMM,LOCID,REGION,DDSO_SA,CLASS_ID,STATE ORDER BY YYYYMM,LOCID,REGION,DDSO_SA,CLASS_ID,STATE".format(ym, ym)
		payload['fm_m'] = month
		payload['fy_m'] = year
		payload['tm_m'] = month
		payload['ty_m'] = year
		print('Requesting year: {}, month: {}...'.format(year, month))
		r = requests.post(api, headers=headers, data=payload)
		soup = bs(r.text, features='lxml')
		table = soup.findAll('table')[0]
		table_body = table.find('tbody')
		rows = table.find_all('tr')
		data = []
		for row in rows:
			cols = row.find_all('td')
			cols = [ele.text.strip() for ele in cols]
			data.append([ele for ele in cols if ele])
		for i in range(3, len(data)):
			row = data[i]
			if 'Total' in row[0]:
				continue
			for k in range(len(contents)):
				contents[k].append(row[k])

# Generate dataframe
df = pd.DataFrame()
for i in range(len(contents)):
	df[columns[i]] = contents[i]
df.to_csv('tracon_data.csv')


