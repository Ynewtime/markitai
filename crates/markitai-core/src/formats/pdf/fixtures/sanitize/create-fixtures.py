from pathlib import Path
import hashlib,json,zlib
root=Path(__file__).resolve().parent
def text(y,value):return f'BT /F1 12 Tf 1 0 0 1 40 {y} Tm ({value}) Tj ET\n'.encode()
def create(name,content,extra=b'',shared=False,image=False):
    objects=[b'<< /Type /Catalog /Pages 2 0 R >>',b'<< /Type /Pages /Count 1 /Kids [3 0 R] /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> /ExtGState << /Zero 5 0 R /Opaque 6 0 R >> '+(b'/XObject << /Shared 8 0 R >> ' if shared else b'/XObject << /Picture 8 0 R >> ' if image else b'')+b'>> >>',b'<< /Type /Page /Parent 2 0 R /Contents 7 0 R >>',b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',b'<< /Type /ExtGState /ca 0 >>',b'<< /Type /ExtGState /ca 1 /CA 1 >>']
    compressed=zlib.compress(content);objects.append(f'<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n'.encode()+compressed+b'\nendstream')
    if shared:objects.append(f'<< /Type /XObject /Subtype /Form /BBox [0 0 612 792] /Length {len(extra)} >>\nstream\n'.encode()+extra+b'\nendstream')
    if image:
        if image=='scan':
            rgb=bytearray([245])*612*792
            for value,baseline in scan_lines:
                for y in range(baseline,baseline+8):
                    for x in range(72,min(612,int(72+len(value)*5.5)),3):rgb[(791-y)*612+x]=20
            compressed=zlib.compress(rgb);description=b'/Width 612 /Height 792 /ColorSpace /DeviceGray'
        else:compressed=zlib.compress(bytes([40,80,160])*128*128);description=b'/Width 128 /Height 128 /ColorSpace /DeviceRGB'
        objects.append(b'<< /Type /XObject /Subtype /Image '+description+f' /BitsPerComponent 8 /Filter /FlateDecode /Length {len(compressed)} >>\nstream\n'.encode()+compressed+b'\nendstream')
    pdf=bytearray(b'%PDF-1.7\n%\xe2\xe3\xcf\xd3\n');offsets=[0]
    for i,obj in enumerate(objects,1):offsets.append(len(pdf));pdf.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(pdf);pdf.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:pdf.extend(f'{offset:010} 00000 n \n'.encode())
    pdf.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode());path=root/name;path.write_bytes(pdf)
    return {'file':name,'bytes':len(pdf),'sha256':hashlib.sha256(pdf).hexdigest(),'provenance':'Authored minimal PDF operators; no external or personal input.'}
intro=text(740,'The visible harbour report contains readable ordinary paragraphs.')
repeated='Shared words occur visibly in the ordinary report.'
content=intro+text(700,repeated)+b'q 1 g '+text(660,repeated)+b'Q q /Zero gs '+text(620,'TRANSPARENT UNIQUE HIDDEN TOKEN')+b'Q BT /F1 0.5 Tf 1 0 0 1 40 560 Tm (TINY UNIQUE HIDDEN TOKEN) Tj ET\n'+text(520,'The final visible paragraph must remain in every policy.')
records=[create('policy-text.pdf',content),create('policy-assets.pdf',content+b'q 160 0 0 160 350 250 cm /Picture Do Q',image=True),create('policy-form.pdf',intro+b'q /Zero gs /Shared Do Q q 1 0 0 1 0 -80 cm /Shared Do Q '+text(500,'The last visible paragraph follows both Form invocations.'),text(680,'SHARED FORM TEXT MUST REMAIN ONCE'),shared=True),create('policy-white-on-black.pdf',intro+b'q 0 g 30 600 540 80 re f 1 g '+text(640,'VISIBLE WHITE ON BLACK PANEL')+b'Q '+text(540,'The ordinary report continues below the highlighted panel.'))]
scan_lines=[('Harbour authority quarterly report',700),('The harbour handled more ships this quarter than in any',680),('quarter before it, and the new berth opened in March.',664),('Repairs to the north pier are planned for the autumn.',648),('Dredging of the channel finished two weeks early.',632),('Fees for small craft stay as they were last year.',616)]
layer=b'q 612 0 0 792 0 0 cm /Picture Do Q\n'+b''.join(f'BT 3 Tr 1 0 0 1 72 {y} Tm /F1 11 Tf ({value}) Tj ET\n'.encode() for value,y in scan_lines)
records.append(create('policy-searchable-scan.pdf',layer,image='scan'))
(root/'author-fixtures.json').write_text(json.dumps({'fixtures':records},indent=2)+'\n');print(json.dumps(records,indent=2))
