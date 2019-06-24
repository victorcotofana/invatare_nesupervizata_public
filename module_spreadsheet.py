import datetime
import gspread
import json
from authlib.client import AssertionSession

SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']


def create_assertion_session(conf_file, scopes, subject=None):
    with open(conf_file, 'r') as f:
        conf = json.load(f)

    token_url = conf['token_uri']
    issuer = conf['client_email']
    key = conf['private_key']
    key_id = conf.get('private_key_id')

    header = {'alg': 'RS256'}
    if key_id:
        header['kid'] = key_id

    # google puts scope in payload
    claims = {'scope': ' '.join(scopes)}
    return AssertionSession(
        grant_type=AssertionSession.JWT_BEARER_GRANT_TYPE,
        token_url=token_url,
        issuer=issuer,
        audience=token_url,
        claims=claims,
        subject=subject,
        key=key,
        header=header,
    )


def get_hours_passed(past_time):
    current_time = datetime.datetime.now()
    duration = current_time - past_time

    duration_in_hours = duration.total_seconds() / 3600
    return duration_in_hours


def write_line(start_time, no_epoch, is_model_saved):
    # as you may have guessed I live in a different time-zone
    now = datetime.datetime.now() + datetime.timedelta(hours=3)
    hours_passed = get_hours_passed(start_time)

    session = create_assertion_session('module_spreadsheet_client_secret.json', SCOPES)
    gspread_client = gspread.Client(None, session)
    sheet = gspread_client.open('unsupervised_video_learning_logfile').sheet1

    # date_stamp | time_passed | no_epoch | saved_model
    row_to_insert = [str(now), str(hours_passed), str(no_epoch), str(is_model_saved)]

    # insert a new row at the top
    sheet.insert_row(row_to_insert)


if __name__ == '__main__':
    write_line(datetime.datetime.now(), 'THIS IS', 'A TEST')
