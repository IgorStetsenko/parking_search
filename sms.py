from twilio.rest import Client

class Sms_delivery():

    def pull_sms(self):
        '''The function sms delivery'''
        # Twilio account details
        twilio_account_sid = ''
        twilio_auth_token = ''
        twilio_source_phone_number = '+'

        # Create a Twilio client object instance
        client = Client(twilio_account_sid, twilio_auth_token)

        # Send an SMS
        message = client.messages.create(
            body="Seat is free!!!",
            from_=twilio_source_phone_number,
            to=""
        )

