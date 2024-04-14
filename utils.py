import os

def push_to_git():
    os.system('git config --global user.name "pollyjuice74"')
    os.system('git remote set-url origin https://pollyjuice74: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@github.com/pollyjuice74/DDECC.git')
    
    os.system('git add .')
    os.system('git commit -m "Add trained model weights"')
    os.system('git push origin main')
