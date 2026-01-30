contract X{
uint256 private a;
mapping(address=>uint256) private b;
function f(uint256 x) external returns(uint256){
a^=x+(a<<1);
b[msg.sender]=(b[msg.sender]+a)^x;
return uint256(keccak256(abi.encodePacked(a,b[msg.sender],block.number)));
}
function g() external view returns(uint256){
return (a>>2)^b[msg.sender];
}
}

