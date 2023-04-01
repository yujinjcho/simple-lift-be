
# prior
# user uploads to s3
# on completion, create video_upload record
    # user_id, lift_id, status (pending)

# fetch video_upload for lift on detail page
# if complete, show button to download/play lift / also add option to download (if easy)

def run_pipeline():
    # get pending video uploads

    # for each pending video
        # download video locally (data/user/lift.ext)
        # process video (data/processed/user/lift.ext)
        # upload to s3 'processed/user/lift_id.ext'
        # delete original upload
        # update video_upload (status:complete)

    return


# Steps 1:
# create video_upload table
# frontend: after upload, create a record
# run pipeline: integrate with supabase
# run pipeline: get pending video_upload