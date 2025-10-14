"""
QR Code scanner for Aadhaar cards - Extract data with 100% accuracy
"""

import logging
import re
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Check for QR code library availability
try:
    from pyzbar.pyzbar import decode
    from pyzbar.pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    logger.warning("pyzbar not available - QR code scanning disabled")


class AadhaarQRScanner:
    """
    Scanner for Aadhaar QR codes
    
    Aadhaar QR codes contain XML data with all fields:
    - Aadhaar number
    - Name
    - Date of birth
    - Gender
    - Address (complete)
    - Photo (base64 encoded)
    """
    
    def __init__(self):
        self.available = PYZBAR_AVAILABLE
    
    def is_available(self) -> bool:
        """Check if QR scanner is available"""
        return self.available
    
    def scan_aadhaar_qr(self, image: np.ndarray) -> Optional[Dict]:
        """
        Scan Aadhaar QR code from image
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary with extracted fields or None if no QR found
        """
        if not self.available:
            logger.warning("QR scanner not available - install pyzbar")
            return None
        
        try:
            # Decode QR codes from image
            qr_codes = decode(image, symbols=[ZBarSymbol.QRCODE])
            
            if not qr_codes:
                logger.debug("No QR codes found in image")
                return None
            
            logger.info(f"Found {len(qr_codes)} QR code(s) in image")
            
            # Process each QR code (Aadhaar cards typically have 1-2 QR codes)
            for qr in qr_codes:
                try:
                    # Decode QR data
                    qr_data = qr.data.decode('utf-8')
                    
                    # Parse Aadhaar QR data
                    fields = self._parse_aadhaar_qr_data(qr_data)
                    
                    if fields:
                        logger.info("Successfully extracted data from Aadhaar QR code")
                        return fields
                        
                except Exception as e:
                    logger.debug(f"Failed to parse QR code: {e}")
                    continue
            
            logger.warning("QR codes found but none contained valid Aadhaar data")
            return None
            
        except Exception as e:
            logger.error(f"QR scanning failed: {e}")
            return None
    
    def _parse_aadhaar_qr_data(self, qr_data: str) -> Optional[Dict]:
        """
        Parse Aadhaar QR code data
        
        Aadhaar QR codes use XML format with specific structure
        
        Args:
            qr_data: Raw QR code data string
            
        Returns:
            Dictionary with parsed fields or None
        """
        try:
            # Aadhaar QR codes are XML formatted
            # Try to parse as XML
            if qr_data.startswith('<?xml') or '<PrintLetterBarcodeData' in qr_data:
                return self._parse_aadhaar_xml(qr_data)
            
            # Some Aadhaar QR codes use pipe-separated format
            elif '|' in qr_data:
                return self._parse_aadhaar_pipe_format(qr_data)
            
            else:
                logger.debug("QR data format not recognized as Aadhaar")
                return None
                
        except Exception as e:
            logger.error(f"Failed to parse Aadhaar QR data: {e}")
            return None
    
    def _parse_aadhaar_xml(self, xml_data: str) -> Optional[Dict]:
        """
        Parse XML format Aadhaar QR code
        
        XML structure:
        <PrintLetterBarcodeData>
            <uid>1234567890123</uid>
            <name>John Doe</name>
            <dob>01-01-1990</dob>
            <gender>M</gender>
            <co>S/O: Father Name</co>
            <house>House No</house>
            <street>Street Name</street>
            <lm>Landmark</lm>
            <loc>Locality</loc>
            <vtc>Village/Town/City</vtc>
            <po>Post Office</po>
            <dist>District</dist>
            <subdist>Sub District</subdist>
            <state>State Name</state>
            <pc>123456</pc>
        </PrintLetterBarcodeData>
        """
        try:
            root = ET.fromstring(xml_data)
            
            fields = {}
            
            # Extract UID (Aadhaar number)
            uid = root.findtext('uid')
            if uid:
                # Mask Aadhaar number
                fields['id_number'] = f"XXXX XXXX {uid[-4:]}"
                fields['id_number_full'] = uid  # Store full number separately
            
            # Extract name
            name = root.findtext('name')
            if name:
                fields['name'] = name.strip()
            
            # Extract DOB
            dob = root.findtext('dob')
            if dob:
                # Convert DD-MM-YYYY to YYYY-MM-DD
                fields['dob'] = self._convert_date_format(dob)
            
            # Extract gender
            gender = root.findtext('gender')
            if gender:
                fields['gender'] = 'MALE' if gender.upper() in ['M', 'MALE'] else 'FEMALE'
            
            # Extract address components
            address_parts = []
            
            # Care of (father/mother name)
            co = root.findtext('co')
            if co:
                fields['father_name'] = co.replace('S/O:', '').replace('D/O:', '').replace('C/O:', '').strip()
            
            # Build full address
            house = root.findtext('house')
            if house:
                address_parts.append(house)
            
            street = root.findtext('street')
            if street:
                address_parts.append(street)
            
            lm = root.findtext('lm')  # Landmark
            if lm:
                address_parts.append(lm)
            
            loc = root.findtext('loc')  # Locality
            if loc:
                address_parts.append(loc)
            
            vtc = root.findtext('vtc')  # Village/Town/City
            if vtc:
                address_parts.append(vtc)
                fields['city'] = vtc
            
            dist = root.findtext('dist')  # District
            if dist:
                address_parts.append(dist)
                fields['district'] = dist
            
            state = root.findtext('state')
            if state:
                address_parts.append(state)
                fields['state'] = state
            
            pc = root.findtext('pc')  # Pincode
            if pc:
                address_parts.append(pc)
                fields['pincode'] = pc
            
            # Combine into full address
            if address_parts:
                fields['address'] = ', '.join(filter(None, address_parts))
            
            # Add source metadata
            fields['source'] = 'qr_code'
            fields['confidence'] = 1.0  # QR code data is 100% accurate
            
            logger.info(f"Extracted {len(fields)} fields from Aadhaar QR code")
            return fields
            
        except Exception as e:
            logger.error(f"Failed to parse Aadhaar XML: {e}")
            return None
    
    def _parse_aadhaar_pipe_format(self, pipe_data: str) -> Optional[Dict]:
        """
        Parse pipe-separated format Aadhaar QR code
        
        Format: UID|Name|DOB|Gender|Address|...
        """
        try:
            parts = pipe_data.split('|')
            
            if len(parts) < 5:
                return None
            
            fields = {}
            
            # UID (Aadhaar number)
            if len(parts) > 0 and parts[0].isdigit() and len(parts[0]) == 12:
                uid = parts[0]
                fields['id_number'] = f"XXXX XXXX {uid[-4:]}"
                fields['id_number_full'] = uid
            
            # Name
            if len(parts) > 1:
                fields['name'] = parts[1].strip()
            
            # DOB
            if len(parts) > 2:
                fields['dob'] = self._convert_date_format(parts[2])
            
            # Gender
            if len(parts) > 3:
                gender = parts[3].upper()
                fields['gender'] = 'MALE' if gender in ['M', 'MALE'] else 'FEMALE'
            
            # Address
            if len(parts) > 4:
                fields['address'] = parts[4].strip()
            
            fields['source'] = 'qr_code'
            fields['confidence'] = 1.0
            
            return fields
            
        except Exception as e:
            logger.error(f"Failed to parse pipe format: {e}")
            return None
    
    def _convert_date_format(self, date_str: str) -> str:
        """
        Convert date from DD-MM-YYYY or DD/MM/YYYY to YYYY-MM-DD
        """
        try:
            # Handle DD-MM-YYYY or DD/MM/YYYY
            if '-' in date_str:
                parts = date_str.split('-')
            elif '/' in date_str:
                parts = date_str.split('/')
            else:
                return date_str
            
            if len(parts) == 3:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            
            return date_str
            
        except Exception:
            return date_str


# Singleton instance
_qr_scanner: Optional[AadhaarQRScanner] = None


def get_qr_scanner() -> AadhaarQRScanner:
    """Get global QR scanner instance"""
    global _qr_scanner
    if _qr_scanner is None:
        _qr_scanner = AadhaarQRScanner()
    return _qr_scanner


def scan_aadhaar_qr(image: np.ndarray) -> Optional[Dict]:
    """
    Convenience function to scan Aadhaar QR code
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with extracted fields or None
    """
    scanner = get_qr_scanner()
    return scanner.scan_aadhaar_qr(image)


# Test function
def test_qr_scanner():
    """Test QR scanner availability"""
    scanner = get_qr_scanner()
    
    return {
        "available": scanner.is_available(),
        "library": "pyzbar" if PYZBAR_AVAILABLE else "none",
        "message": "QR scanning ready" if scanner.is_available() else "Install pyzbar: pip install pyzbar"
    }


if __name__ == "__main__":
    result = test_qr_scanner()
    print("QR Scanner test:", result)
