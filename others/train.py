# SECSConfig>
#     <Reports>
#         <Report RPTID="1">
#             <DataVariable DVID="1001" Name="StudentName" />
#             <DataVariable DVID="1002" Name="StudentID" />
#         </Report>
#         <Report RPTID="2">
#             <DataVariable DVID="1003" Name="Grade" />
#         </Report>
#     </Reports>
#     <Events>
#         <Event CEID="20">
#             <ReportID>1</ReportID>
#             <ReportID>2</ReportID>
#         </Event>
#     </Events>
# </SECSConfig>



Imports System.Xml.Linq
Imports System.Reflection

Public Class DataVariable
    Public Property DVID As Integer
    Public Property Name As String
End Class

Public Class Report
    Public Property RPTID As Integer
    Public Property DVIDs As List(Of DataVariable)
End Class

Public Class EventDef
    Public Property CEID As Integer
    Public Property ReportIDs As List(Of Integer)
End Class

Public Class Tag
    ' This class will dynamically hold properties corresponding to DVIDs
End Class

Public Class SECSMessageParser
    Private Reports As Dictionary(Of Integer, Report)
    Private Events As Dictionary(Of Integer, EventDef)
    Private TagObject As Tag

    Public Sub New(configFile As String)
        Reports = New Dictionary(Of Integer, Report)()
        Events = New Dictionary(Of Integer, EventDef)()
        TagObject = New Tag()
        LoadConfig(configFile)
    End Sub

    Private Sub LoadConfig(configFile As String)
        Dim xDoc = XDocument.Load(configFile)
        
        ' Load Reports
        For Each xReport In xDoc.Descendants("Report")
            Dim rptID = Convert.ToInt32(xReport.Attribute("RPTID").Value)
            Dim dataVars As New List(Of DataVariable)
            For Each xDataVar In xReport.Descendants("DataVariable")
                Dim dvid = Convert.ToInt32(xDataVar.Attribute("DVID").Value)
                Dim name = xDataVar.Attribute("Name").Value
                dataVars.Add(New DataVariable With {.DVID = dvid, .Name = name})
            Next
            Reports(rptID) = New Report With {.RPTID = rptID, .DVIDs = dataVars}
        Next

        ' Load Events
        For Each xEvent In xDoc.Descendants("Event")
            Dim ceid = Convert.ToInt32(xEvent.Attribute("CEID").Value)
            Dim reportIDs As New List(Of Integer)
            For Each xReportID In xEvent.Descendants("ReportID")
                reportIDs.Add(Convert.ToInt32(xReportID.Value))
            Next
            Events(ceid) = New EventDef With {.CEID = ceid, .ReportIDs = reportIDs}
        Next
    End Sub

    Public Sub ParseMessage(ceid As Integer, transaction As Object)
        If Not Events.ContainsKey(ceid) Then
            Throw New Exception("Unknown CEID")
        End If

        Dim eventDef = Events(ceid)
        For Each rptID In eventDef.ReportIDs
            If Not Reports.ContainsKey(rptID) Then
                Continue For
            End If

            Dim report = Reports(rptID)
            Dim reportData = GetReportData(transaction, rptID)

            For Each dvid In report.DVIDs
                Dim value = GetTransactionValue(reportData, dvid.DVID)
                SetProperty(TagObject, dvid.Name, value)
            Next
        Next
    End Sub

    Private Function GetReportData(transaction As Object, rptID As Integer) As Object
        ' Implement logic to navigate the transaction object to find the data for the specified rptID
        For Each item In transaction.item(1).item(3)
            If item.item(1).value = rptID Then
                Return item.item(2)
            End If
        Next
        Return Nothing
    End Function

    Private Function GetTransactionValue(reportData As Object, dvid As Integer) As Object
        ' Implement logic to extract value from the reportData object based on dvid
        For Each item In reportData.item(2)
            If item.item(1).value = dvid Then
                Return item.item(2).value
            End If
        Next
        Return Nothing
    End Function

    Private Sub SetProperty(obj As Object, propName As String, value As Object)
        Dim prop = obj.GetType().GetProperty(propName)
        If prop Is Nothing Then
            ' Create property dynamically using Reflection.Emit or other techniques
            prop = obj.GetType().DefineProperty(propName, value.GetType())
        End If
        prop.SetValue(obj, value)
    End Sub

    Public ReadOnly Property Tag As Tag
        Get
            Return TagObject
        End Get
    End Property
End Class

Module Module1
    Sub Main()
        ' Define the path to the configuration XML
        Dim configFile As String = "path_to_your_config_file.xml"

        ' Create the parser with the configuration
        Dim parser As New SECSMessageParser(configFile)

        ' Create a sample transaction object
        Dim transaction As New Object() ' Replace with actual transaction object

        ' Parse the message
        parser.ParseMessage(20, transaction)

        ' Output the results
        Dim tag As Tag = parser.Tag
        Dim props = tag.GetType().GetProperties()
        For Each prop In props
            Console.WriteLine($"{prop.Name}: {prop.GetValue(tag)}")
        Next
    End Sub
End Module