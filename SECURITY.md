# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions of vitalDSP:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a Vulnerability

The vitalDSP team takes security bugs seriously. We appreciate your efforts to responsibly disclose your findings and will make every effort to acknowledge your contributions.

### How to Report a Security Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
- **Email**: vital.data@oucru.org
- **Subject**: [SECURITY] Brief description of the vulnerability

### What to Include in Your Report

To help us better understand the nature and scope of the possible issue, please include as much of the following information as possible:

* Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

### What to Expect

After you submit a report, here's what will happen:

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 3 business days.

2. **Investigation**: Our security team will investigate the issue and determine its impact and severity.

3. **Updates**: We will keep you informed about our progress. We may ask for additional information or guidance.

4. **Fix Development**: Once the vulnerability is confirmed, we will work on a fix. The timeline will depend on the complexity and severity of the issue.

5. **Release**: We will release a patch as soon as possible. For critical vulnerabilities, we may issue an emergency release.

6. **Disclosure**: After the fix is released, we will publicly disclose the vulnerability. We will credit you for the discovery unless you prefer to remain anonymous.

## Security Best Practices for Users

When using vitalDSP, we recommend the following security best practices:

### 1. Keep Your Installation Updated

Always use the latest version of vitalDSP to ensure you have the latest security patches:

```bash
pip install --upgrade vitaldsp
```

### 2. Validate Input Data

When processing physiological signals, especially from external sources:

* Validate file formats and sizes before processing
* Sanitize file names and paths to prevent directory traversal attacks
* Set appropriate file size limits to prevent resource exhaustion
* Use secure file upload practices in web applications

### 3. Web Application Security

If deploying the vitalDSP web application:

* Use HTTPS in production environments
* Implement proper authentication and authorization
* Set appropriate CORS policies
* Use environment variables for sensitive configuration
* Regularly update all dependencies
* Implement rate limiting to prevent abuse
* Use secure session management

### 4. Data Privacy

When handling patient or sensitive data:

* Comply with relevant healthcare data regulations (HIPAA, GDPR, etc.)
* Implement proper data encryption at rest and in transit
* Use secure data storage practices
* Implement proper access controls
* Maintain audit logs for data access
* Ensure proper data anonymization when required

### 5. Dependency Management

* Regularly update dependencies to patch known vulnerabilities
* Use virtual environments to isolate dependencies
* Review security advisories for Python packages
* Use tools like `pip-audit` or `safety` to check for known vulnerabilities:

```bash
pip install pip-audit
pip-audit
```

### 6. Machine Learning Model Security

When using ML/DL features:

* Validate model inputs to prevent adversarial attacks
* Use trusted pre-trained models only
* Implement model versioning and validation
* Monitor model predictions for anomalies
* Secure model files with appropriate permissions

## Known Security Considerations

### NumPy and TensorFlow Compatibility

vitalDSP constrains NumPy to version < 2.0 for TensorFlow compatibility. While this is necessary for current TensorFlow versions, be aware that:

* NumPy 1.x may have known vulnerabilities
* We monitor security advisories and will update constraints as TensorFlow adds NumPy 2.x support
* Consider using isolated environments for production deployments

### Web Application Limitations

The included web application is designed for demonstration and development:

* Not hardened for public internet deployment without additional security measures
* Requires additional configuration for production use (authentication, HTTPS, etc.)
* File upload functionality should be restricted in production environments

## Security Update Policy

* **Critical vulnerabilities**: Patch released within 7 days
* **High severity**: Patch released within 30 days
* **Medium/Low severity**: Included in next regular release

## Bug Bounty Program

We currently do not have a bug bounty program. However, we deeply appreciate security researchers who responsibly disclose vulnerabilities and will acknowledge their contributions in our release notes and security advisories.

## Contact

For any security-related questions or concerns, please contact:
- **Email**: vital.data@oucru.org
- **GitHub**: [Report a security vulnerability](https://github.com/Oucru-Innovations/vital-DSP/security/advisories/new)

## Additional Resources

* [OWASP Top Ten](https://owasp.org/www-project-top-ten/)
* [Python Security Best Practices](https://python.readthedocs.io/en/latest/library/security_warnings.html)
* [Healthcare Data Security (HIPAA)](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
* [GDPR Compliance](https://gdpr.eu/)

---

Thank you for helping keep vitalDSP and our users safe!
