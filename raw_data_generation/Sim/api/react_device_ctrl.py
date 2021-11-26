import time
import os
from datetime import datetime, timedelta
#import paho.mqtt.client as paho
import ssl
import logging
import json
import uuid

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class REACTPlatformCommunication:

    def __init__(self, userdata, broker_address, port, timeout=60,
                 cert_loc=os.path.join(os.path.dirname(__file__), "..", "mosq-ca.crt"),
                 data_topic="MID-DATAGATEWAYID001/data", response_topic="MID-DATAGATEWAYID001/response",
                 request_topic="MID-DATAGATEWAYID001/request", device_id="MID-00008484E913D3DB",
                 control_id="SetTemperature"):

        self._userdata = userdata
        self._cert_loc = cert_loc

        self._broker_address = broker_address
        self._port = port
        self._timeout = timeout

        self.data_topic = data_topic
        self.response_topic = response_topic
        self.request_topic = request_topic

        self.device_id = device_id

        self.client = None

        self.control_id = control_id

        self.room_temperature = 22.0
        self.battery_voltage = 47.9
        self.state_of_charge = 1.0

    def on_message(self, client, userdata, message):
        json_message = json.loads(message.payload.decode('utf-8'))
        logging.debug(f"received message = {json_message}")

        try:
            if json_message[0]["deviceId"] == self.device_id or json_message[0]["deviceId"].startswith(self.device_id):
                logging.debug("LAB Device just submitted data!")
                for measurement in json_message:
                    if "measurementId" in measurement:
                        if measurement["measurementId"] in ["sOc", "SOC", "SoC"]:
                            state_of_charge = measurement["value"]
                            logging.info(f"State of charge provided: {state_of_charge}")
                            self.set_soc(state_of_charge)

                    if measurement["deviceId"] == f"{self.device_id}-RoomTemperature":
                        self.room_temperature = measurement["value"]
                        logging.info(f"Room temperature has been submitted. It is {self.room_temperature}.")

        except Exception as e:
            logging.debug("error during reading of message, probably message was not a list")

        # TODO check why this works for midac but not victron
        # Does it have something to do with message formatting?
        # midac sends all values as a list that needs to be decoded.
        # victron throws an error when trying to decode and sends each measurement as own packet.

    @staticmethod
    def on_log(client, userdata, level, buf):
        logging.debug(buf)

    @staticmethod
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logging.debug("connected!")
        else:
            logging.debug("not connected!")

        logging.debug(f"flags: {flags}, rc: {rc}")

    def setup_mqtt_connection(self):  # , on_message=on_message, on_log=on_log, on_connect=on_connect):

        self.client = paho.Client(client_id="AIT-96d05e7cc89947a1827ece46fdcda283")

        self.client.on_connect = self.on_connect
        self.client.on_log = self.on_log
        self.client.on_message = self.on_message  # (comm_class=self, device_id=self.device_id)

        self.client.tls_set(self._cert_loc, tls_version=ssl.PROTOCOL_TLSv1_2)
        self.client.tls_insecure_set(False)

        self.client.username_pw_set(username=self._userdata["username"], password=self._userdata["password"])

    def connect(self):

        logging.debug(f"connecting to broker: {self._broker_address}, port: {self._port}, timeout: {self._timeout}")
        self.client.connect(self._broker_address, self._port, self._timeout)

    def subscribe(self):

        self.client.subscribe(self.data_topic, qos=0)
        # self.client.subscribe(self.response_topic, qos=0)
        self.client.subscribe(self.request_topic, qos=0)

    def publish(self, msg):

        self.client.publish(self.request_topic, msg, qos=0)

    def set_soc(self, soc):

        self.state_of_charge = soc

    def set_value(self, value):

        if self.device_id.startswith("MID"):
            if self.state_of_charge < 20.0 and value > 0:
                logging.info("Midac SoC critical - not setting discharge value! setting 0 value instead.")
                value = -10

        elif self.device_id.startswith("VIC"):
            if self.state_of_charge < 10.0 or self.battery_voltage < 46.0 and value > 0:
                logging.info(
                    f"Victron SoC ({self.state_of_charge}) or battery voltage ({self.battery_voltage}) critical "
                    " - not setting discharge value! setting 0 value instead.")
                value = 0

        elif self.device_id.startswith("MEL"):
            value = value

        msg = self.format_message(req_value=value)

        logging.info(msg)

        self.publish(msg)

    def format_message(self, req_value, request_id="aitlabreq"):
        msg_dict = {"timestamp": round(time.mktime(datetime.timetuple(datetime.now())) * 1000),
                    "value": req_value,
                    "deviceId": self.device_id,
                    "requestId": f"{request_id}_{uuid.uuid4()}",
                    "controlId": self.control_id}
        # {"requestId": "ait-request-123", "value":true, "deviceId": "MEL-27597", "controlId": "Power"}
        if self.device_id.startswith("MID"):
            return json.dumps([msg_dict])
        else:
            return json.dumps(msg_dict)

    def start(self):

        self.client.loop_start()

    def stop(self):

        self.client.loop_stop()


if __name__ == '__main__':

    logging.debug("Starting comm setup")

    react_comm = REACTPlatformCommunication(
        userdata={"username": "react_ait", "password": "8769n2v86co8976ov9746ov9voo"},
        broker_address="147.91.50.80",
        port=8883, timeout=60)

    react_comm.setup_mqtt_connection()

    logging.debug("Trying to connect")

    react_comm.connect()

    logging.debug("Subscribing..")

    react_comm.subscribe()

    logging.debug("Starting loop..!")

    react_comm.start()

    while True:
        time.sleep(15)

    # power_ramp = list(range(-50, -450, -50))
    #
    # for power_list in [power_ramp, list(reversed(power_ramp))]:
    #
    #     for power in power_list:
    #
    #         midac_comm.set_value(power)
    #
    #         time.sleep(60)
