import datetime
import pickle
import os.path

from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build

# if modifying these scopes, delete the file token.pickle
SCOPES_OLD = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'https://www.googleapis.com/auth/drive']
SCOPES = ['https://www.googleapis.com/auth/drive']

# unsupervised_video_learning_models : folder ID in Google Drive
FOLDER_ID = '161AOGyTKp2Y0Zn5wUWjLjc7tswyebz7I'


def authenticate_user():
    if not os.path.exists('token.pickle'):
        flow = InstalledAppFlow.from_client_secrets_file('module_drive_credentials.json', SCOPES)
        creds = flow.run_local_server(port=8081)

        # save the credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)


def update_or_create_file(file_id, file_name, mimetype):
    creds = None

    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # check if the token is still valid. If not, refersh it without asking the user to login
    if creds and not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    drive_service = build('drive', 'v3', credentials=creds)

    now = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
    new_file_name = now + '_' + file_name

    media = MediaFileUpload(file_name, mimetype=mimetype)

    # if file_id is None then create, if not None then update. Google Drive takes care of the versions
    if file_id is None:
        # create the file
        file_metadata = {'name': new_file_name, 'parents': [FOLDER_ID]}
        file = drive_service.files().create(body=file_metadata,
                                            media_body=media).execute()
    else:
        # update the file
        file_metadata = {'name': new_file_name}
        file = drive_service.files().update(fileId=file_id,
                                            addParents=FOLDER_ID,
                                            body=file_metadata,
                                            media_body=media).execute()
    return file.get('id')


if __name__ == '__main__':
    authenticate_user()
