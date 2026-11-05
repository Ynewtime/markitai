; Markitai Web's additions to the Windows installer that mygo build writes
; (windows.installerInclude of mygo.json). mygo includes this file after
; MUI2.nsh, the pages and the languages, and before its own sections, so
; the sections here run first; it runs customInit in .onInit and
; customInstall at the end of the install.
;
; - A folder that holds files, and is not Markitai Web's, is never the
;   install folder: the uninstaller removes the install folder with all it
;   holds (RMDir /r of mygo's script), so the user's files would go with it.
;   The directory page refuses it, and a silent install (/S /D=...) fails.
; - An installed Markitai Web is updated in its folder: an install
;   elsewhere would leave it there, still running, with nothing to
;   uninstall it. The directory page keeps that folder, and a silent
;   install elsewhere fails; moving it takes an uninstall first.
; - A running Markitai Web is asked through its single-instance lock
;   (--is-running), told what quitting it means, and quit at once
;   (--quit-now) when the install or the uninstall starts, so that one
;   started while the wizard was open is quit too, and leaving the wizard
;   quits none: mygo's own close, CloseMainWindow and then Stop-Process
;   after 10 s, would show the app's question about unfinished conversions,
;   whose answer the forced stop then ignores.

!define MARKITAI_WEB_EXE "Markitai Web.exe"
!define MARKITAI_WEB_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\dev.markitai.web"

LangString MARKITAI_WEB_FOREIGN ${LANG_ENGLISH} "This folder holds other files. Choose an empty folder, or the one Markitai Web is installed in: uninstalling removes the install folder with everything in it."
LangString MARKITAI_WEB_INSTALLED ${LANG_ENGLISH} "Markitai Web is installed in this folder, and Setup updates it here. To install it in another folder, uninstall it first."
LangString MARKITAI_WEB_RUNNING ${LANG_ENGLISH} "Markitai Web is running. Setup will quit it, and conversions it has not finished will be stopped."
LangString MARKITAI_WEB_RUNNING_UN ${LANG_ENGLISH} "Markitai Web is running. Uninstalling will quit it, and conversions it has not finished will be stopped."
!ifdef LANG_SIMPCHINESE
LangString MARKITAI_WEB_FOREIGN ${LANG_SIMPCHINESE} "这个文件夹里有其他文件。请选择一个空文件夹，或 Markitai Web 已安装的文件夹：卸载时会删除安装文件夹及其中的所有内容。"
LangString MARKITAI_WEB_INSTALLED ${LANG_SIMPCHINESE} "Markitai Web 已安装在这个文件夹，安装程序会在这里更新它。要安装到其他文件夹，请先卸载它。"
LangString MARKITAI_WEB_RUNNING ${LANG_SIMPCHINESE} "Markitai Web 正在运行。安装程序会退出它，尚未完成的转换会被停止。"
LangString MARKITAI_WEB_RUNNING_UN ${LANG_SIMPCHINESE} "Markitai Web 正在运行。卸载会退出它，尚未完成的转换会被停止。"
!endif
!ifdef LANG_JAPANESE
LangString MARKITAI_WEB_FOREIGN ${LANG_JAPANESE} "このフォルダーには他のファイルがあります。空のフォルダーか、Markitai Web がインストールされているフォルダーを選んでください。アンインストールすると、インストール先のフォルダーは中身ごと削除されます。"
LangString MARKITAI_WEB_INSTALLED ${LANG_JAPANESE} "Markitai Web はこのフォルダーにインストールされていて、セットアップはここで更新します。別のフォルダーにインストールするには、先にアンインストールしてください。"
LangString MARKITAI_WEB_RUNNING ${LANG_JAPANESE} "Markitai Web が実行中です。セットアップは Markitai Web を終了し、完了していない変換は停止されます。"
LangString MARKITAI_WEB_RUNNING_UN ${LANG_JAPANESE} "Markitai Web が実行中です。アンインストールすると Markitai Web は終了し、完了していない変換は停止されます。"
!endif

Var MarkitaiWebIntro

; Leaves on the stack the folder Markitai Web is installed in, the one its
; uninstall entry names when its program is there, else "".
Function MarkitaiWebInstalled
  Push $0
  ReadRegStr $0 HKCU "${MARKITAI_WEB_KEY}" "InstallLocation"
  StrCmp $0 "" +3
  IfFileExists "$0\${MARKITAI_WEB_EXE}" +2
  StrCpy $0 ""
  Exch $0
FunctionEnd

; Leaves 1 on the stack when $INSTDIR holds files and is not Markitai Web's
; folder, else 0. Markitai Web's is the one its uninstall entry names, or
; one holding both its program and its uninstaller.
!macro MarkitaiWebForeign un
Function ${un}MarkitaiWebForeign
  Push $0
  Push $1
  ReadRegStr $0 HKCU "${MARKITAI_WEB_KEY}" "InstallLocation"
  StrCmp $0 "" +3
  StrCmp $0 $INSTDIR ours
  StrCmp $0 "$INSTDIR\" ours
  IfFileExists "$INSTDIR\${MARKITAI_WEB_EXE}" 0 +2
  IfFileExists "$INSTDIR\Uninstall.exe" ours
  ClearErrors
  FindFirst $0 $1 "$INSTDIR\*"
  IfErrors empty
  next:
    StrCmp $1 "" closed
    StrCmp $1 "." more
    StrCmp $1 ".." more
    FindClose $0
    Goto foreign
  more:
    FindNext $0 $1
    Goto next
  closed:
  FindClose $0
  empty:
  ours:
  Pop $1
  Pop $0
  Push 0
  Return
  foreign:
  Pop $1
  Pop $0
  Push 1
FunctionEnd
!macroend
!insertmacro MarkitaiWebForeign ""

; Quits a running Markitai Web once the user agrees, asking with the
; message on the stack; a silent run agrees. Aborts when the user does not.
; Only the app that customInstall registered in $INSTDIR is asked: a build
; from before these requests would take them for a launch, open its
; window and keep the installer waiting; mygo's own close quits that one.
!macro MarkitaiWebQuit un
Function ${un}MarkitaiWebQuit
  Exch $0
  Push $1
  Push $2
  IfFileExists "$INSTDIR\${MARKITAI_WEB_EXE}" 0 done
  ReadRegDWORD $1 HKCU "${MARKITAI_WEB_KEY}" "QuitsOnRequest"
  StrCmp $1 "1" 0 done
  ReadRegStr $1 HKCU "${MARKITAI_WEB_KEY}" "InstallLocation"
  StrCmp $1 $INSTDIR 0 done
  ExecWait '"$INSTDIR\${MARKITAI_WEB_EXE}" --is-running' $1
  StrCmp $1 "2" 0 done
  MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION $0 /SD IDOK IDOK quit
  Pop $2
  Pop $1
  Pop $0
  Abort
  quit:
  ExecWait '"$INSTDIR\${MARKITAI_WEB_EXE}" --quit-now'
  ; The app stops its server first; mygo's close after this stops what is
  ; left after 10 s.
  StrCpy $2 0
  wait:
    ExecWait '"$INSTDIR\${MARKITAI_WEB_EXE}" --is-running' $1
    StrCmp $1 "2" 0 done
    IntOp $2 $2 + 1
    IntCmp $2 40 done
    Sleep 250
    Goto wait
  done:
  Pop $2
  Pop $1
  Pop $0
FunctionEnd
!macroend
!insertmacro MarkitaiWebQuit ""
!insertmacro MarkitaiWebQuit "un."

; An update goes to the installed folder, the one the wizard shows; a
; silent install to another (/D=...) fails, as one to a foreign folder.
!macro customInit
  ; NSIS ends the folder it reads from the uninstall entry
  ; (InstallDirRegKey) with a backslash, and the directory page of a
  ; reinstall shows the variable as it is. $INSTDIR reads without it, so
  ; assigning it to itself drops it; the page gives a drive's root its own.
  StrCpy $INSTDIR $INSTDIR
  Call MarkitaiWebInstalled
  Pop $0
  StrCmp $0 "" markitai_web_new
  StrCmp $0 $INSTDIR markitai_web_ok
  IfSilent markitai_web_refuse
  StrCpy $INSTDIR $0
  Goto markitai_web_ok
  markitai_web_new:
  IfSilent 0 markitai_web_ok
  Call MarkitaiWebForeign
  Pop $0
  StrCmp $0 "1" 0 markitai_web_ok
  markitai_web_refuse:
  SetErrorLevel 2
  Quit
  markitai_web_ok:
!macroend

; The directory page. An installed Markitai Web's folder stays: the field
; is read-only, Browse is off, and the page's text says why. Otherwise Next
; stays off for a folder that is not Markitai Web's and holds files, and
; the page's text says why; it keeps the page's own text, read the first
; time, for the other folders.
Function .onVerifyInstDir
  Push $0
  Push $1
  Push $2
  FindWindow $2 "#32770" "" $HWNDPARENT
  GetDlgItem $0 $2 1006
  StrCmp $MarkitaiWebIntro "" 0 +3
  System::Call 'user32::GetWindowTextW(p r0, w .r1, i ${NSIS_MAX_STRLEN})'
  StrCpy $MarkitaiWebIntro $1
  Call MarkitaiWebInstalled
  Pop $1
  StrCmp $1 "" new
  GetDlgItem $1 $2 1019
  SendMessage $1 ${EM_SETREADONLY} 1 0
  GetDlgItem $1 $2 1001
  EnableWindow $1 0
  SendMessage $0 ${WM_SETTEXT} 0 "STR:$(MARKITAI_WEB_INSTALLED)"
  Goto ok
  new:
  Call MarkitaiWebForeign
  Pop $1
  StrCmp $1 "1" foreign
  SendMessage $0 ${WM_SETTEXT} 0 "STR:$MarkitaiWebIntro"
  ok:
  Pop $2
  Pop $1
  Pop $0
  Return
  foreign:
  SendMessage $0 ${WM_SETTEXT} 0 "STR:$(MARKITAI_WEB_FOREIGN)"
  Pop $2
  Pop $1
  Pop $0
  Abort
FunctionEnd

; The uninstaller runs from a copy of itself, which it hands its folder
; with a backslash added (_?=$INSTDIR\), and the confirm page shows the
; variable as it is; assigning it to itself drops it (customInit).
Function un.onInit
  StrCpy $INSTDIR $INSTDIR
FunctionEnd

; Sections run in the order they are written: these before mygo's, which
; quit the app their own way. They start once the user chose Install or
; Uninstall. The requests to the app would show under the progress bar as
; "Execute: ...".
Section "-MarkitaiWebQuit"
  SetDetailsPrint none
  Push $(MARKITAI_WEB_RUNNING)
  Call MarkitaiWebQuit
  SetDetailsPrint lastused
SectionEnd

Section "-un.MarkitaiWebQuit"
  SetDetailsPrint none
  Push $(MARKITAI_WEB_RUNNING_UN)
  Call un.MarkitaiWebQuit
  SetDetailsPrint lastused
SectionEnd

; The app installed now answers --is-running and --quit-now
; (MarkitaiWebQuit).
!macro customInstall
  WriteRegDWORD HKCU "${MARKITAI_WEB_KEY}" "QuitsOnRequest" 1
!macroend
