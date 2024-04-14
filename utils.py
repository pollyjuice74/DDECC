import os

def push_to_git():
    os.system('git config --global user.name "pollyjuice74"')
    os.system('git remote set-url origin https://pollyjuice74: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@github.com/pollyjuice74/DDECC.git')
    
    os.system('git add .')
    os.system('git commit -m "Add trained model weights"')
    os.system('git push origin main')
