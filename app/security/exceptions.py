from app.exceptions.errors import AppException, ModelGatewayError


class SecurityClassifierError(AppException):
    status_code = 502
    code = 'SECURITY_CLASSIFIER_ERROR'


class SecurityClassifierTimeoutError(SecurityClassifierError):
    status_code = 504
    code = 'SECURITY_CLASSIFIER_TIMEOUT'


class SecurityClassifierUnavailableError(SecurityClassifierError):
    status_code = 503
    code = 'SECURITY_CLASSIFIER_UNAVAILABLE'


class SecurityClassificationResponseError(SecurityClassifierError):
    code = 'SECURITY_CLASSIFIER_RESPONSE_ERROR'


class SecurityClassifierTransportError(SecurityClassifierError):
    code = 'SECURITY_CLASSIFIER_TRANSPORT_ERROR'

    @classmethod
    def from_model_gateway(cls, exc: ModelGatewayError) -> 'SecurityClassifierTransportError':
        return cls(str(exc), status_code=exc.status_code)