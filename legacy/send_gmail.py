#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Jeromie Kirchoff
# Created Date: Mon Aug 02 17:46:00 PDT 2018
# =============================================================================
# Imports
# =============================================================================
import smtplib
import os
from enum import Enum


class Mode(Enum):
    TRAIN = 0
    INFERENCE = 1
    DATAMAKER = 2

# =============================================================================
# SET EMAIL LOGIN REQUIREMENTS
# =============================================================================


def send_gmail(experiment, mode):
    if not os.path.isfile('mail_cfg.txt'):
        return
    else:
        with open("mail_cfg.txt", "r") as f:
            text = f.readlines()
            receiver_email = text[0][:-1]
            #password = text[1]
            gmail_app_password = text[2]

    gmail_user = 'gipmed.python@gmail.com'

    # =============================================================================
    # SET THE INFO ABOUT THE SAID EMAIL
    # =============================================================================
    sent_from = gmail_user
    sent_to = receiver_email

    if mode == Mode.TRAIN:
        sent_subject = 'Subject: finished running experiment ' + str(experiment)
    elif mode == Mode.INFERENCE:
        sent_subject = 'Subject: finished inference for experiment ' + str(experiment)
    elif mode == Mode.DATAMAKER:
        sent_subject = 'Subject: finished running data maker'
    else:
        return

    email_text = sent_subject

    # =============================================================================
    # SEND EMAIL OR DIE TRYING!!!
    # Details: http://www.samlogic.net/articles/smtp-commands-reference.htm
    # =============================================================================

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_app_password)
        server.sendmail(sent_from, sent_to, email_text)
        server.close()

        print('Email sent!')
    except Exception as exception:
        print("Error: %s!\n\n" % exception)